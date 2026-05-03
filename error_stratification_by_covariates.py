'''
NeuRIPS 2026: Error Stratification by Covariates

Evaluates whether BrainGenFlow predicted trajectories maintain group differences
between MCI progressors and non-progressors.

Prediction file format (fold files)
------------------------------------
Filename : trajectory_<PTID>_fold<N>.csv
Columns  : ROI, Month_0 … Month_69
Rows     : 290 = 145 ROIs × 2 types, alternating:
             <roi_idx>_real  – sparse, actual observed values (NaN where unobserved)
             <roi_idx>_gen   – dense, model-predicted trajectory every month

ROI mapping
-----------
hmuse_list.npy[i] gives the MUSE atlas ROI number for list index i.
Merged-DataFrame column names are 'MUSE_<roi_number>'.

Analysis
--------
For each subject, an OLS slope is computed per ROI from the _gen (predicted)
trajectory across Month_0…Month_69.  The resulting wide feature matrix
(subjects × 145 ROIs) is merged with MCI progression labels and tested
ROI-by-ROI with a Mann-Whitney U test (BH FDR correction, α = 0.05).
'''

import os
import re
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PRED_DIR   = Path('./predictions')
COV_FILE   = Path('./longitudinal_covariates_allstudies.csv')
HMUSE_FILE = Path('./hmuse_list.npy')
OUT_DIR    = Path('./staging_analysis')

N_ROIS     = 145
N_MONTHS   = 70   # Month_0 … Month_69

MCI_DIAGNOSES = {
    'MCI',
    'MCI Amnestic Single Domain (Petersen Criteria)',
    'MCI Amnestic Plus Other (Petersen Criteria)',
    'MCI Non-Amnestic Single Domain (Petersen Criteria)',
    'MCI Non-Amnestic Multiple Domains (Petersen Criteria)',
    'MCI - mixed vascular & AD',
}
AD_DIAGNOSES = {
    'AD', 'Dementia', 'AD Dementia', 'AD patient',
    'AD Probable (NINCDS/ADRDA)', 'AD dem w/depresss  not contribut',
    'uncertain  possible NON AD dem', 'Dementia NOS',
}


# ---------------------------------------------------------------------------
# ROI label mapping via hmuse_list.npy
# ---------------------------------------------------------------------------
def build_roi_labels(hmuse_path: Path) -> list:
    """
    Returns a list of 145 column name strings.
    If hmuse_list.npy exists: 'MUSE_<atlas_number>'
    Otherwise: 'ROI_<list_index>'
    """
    if hmuse_path.exists():
        hmuse = np.load(str(hmuse_path), allow_pickle=True)
        labels = [f'MUSE_{int(hmuse[i])}' for i in range(N_ROIS)]
        print(f"  Loaded hmuse_list.npy — ROI labels: {labels[:5]} … {labels[-3:]}")
    else:
        labels = [f'ROI_{i}' for i in range(N_ROIS)]
        print(f"  hmuse_list.npy not found — using list-index labels")
    return labels


# ---------------------------------------------------------------------------
# Load fold prediction files and compute OLS slopes from _gen trajectories
# ---------------------------------------------------------------------------
def load_predicted_slopes(pred_dir: Path, roi_labels: list) -> pd.DataFrame:
    """
    For every trajectory_<PTID>_fold<N>.csv, extract the _gen (predicted)
    rows and compute one OLS slope per ROI over Month_0…Month_69.

    Returns a wide DataFrame: [PTID, MUSE_4, MUSE_11, …, MUSE_207]
    One row per subject (fold files: one file per subject).
    """
    fold_pat  = re.compile(r'^trajectory_(.+)_fold(\d+)\.csv$')
    csv_files = sorted(pred_dir.glob('trajectory_*_fold*.csv'))
    print(f"  Found {len(csv_files)} fold trajectory files")

    month_cols  = [f'Month_{m}' for m in range(N_MONTHS)]
    month_range = np.arange(N_MONTHS, dtype=float)

    rows = []
    skipped = 0
    for fpath in csv_files:
        m = fold_pat.match(fpath.name)
        if not m:
            skipped += 1
            continue
        ptid = m.group(1)

        try:
            df = pd.read_csv(fpath)
        except Exception:
            skipped += 1
            continue

        # Extract _gen rows only
        gen_mask = df['ROI'].str.endswith('_gen')
        gen_df   = df[gen_mask].copy()
        gen_df['roi_idx'] = gen_df['ROI'].str.replace('_gen', '', regex=False).astype(int)
        gen_df = gen_df.set_index('roi_idx')

        # Verify month columns are present
        avail_months = [c for c in month_cols if c in gen_df.columns]
        if len(avail_months) < 2:
            skipped += 1
            continue

        t = np.array([int(c.split('_')[1]) for c in avail_months], dtype=float)

        roi_slopes = {'PTID': ptid}
        for list_idx, label in enumerate(roi_labels):
            if list_idx not in gen_df.index:
                roi_slopes[label] = np.nan
                continue
            vals = gen_df.loc[list_idx, avail_months].values.astype(float)
            if np.isnan(vals).any() or len(vals) < 2:
                roi_slopes[label] = np.nan
            else:
                roi_slopes[label] = np.polyfit(t, vals, 1)[0]

        rows.append(roi_slopes)

    slopes_df = pd.DataFrame(rows)
    print(f"  Computed slopes for {len(slopes_df)} subjects, "
          f"{slopes_df.shape[1] - 1} ROIs  (skipped {skipped} files)")
    return slopes_df


# ---------------------------------------------------------------------------
# Build MCI progression labels
# ---------------------------------------------------------------------------
def build_progression_labels(cov_file: Path) -> pd.DataFrame:
    """
    Returns [PTID, is_progressor] for subjects whose first diagnosis is MCI.
    is_progressor = 1 if any later visit carries an AD/Dementia label.
    """
    cov = pd.read_csv(str(cov_file))
    records = []
    for ptid, grp in cov.groupby('PTID'):
        grp   = grp.sort_values('Time')
        diags = grp['Diagnosis'].tolist()
        if diags[0] not in MCI_DIAGNOSES:
            continue
        is_prog = int(any(d in AD_DIAGNOSES for d in diags[1:]))
        records.append({'PTID': ptid, 'is_progressor': is_prog})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# ROI-level Mann-Whitney U tests + effect size
# ---------------------------------------------------------------------------
def rank_biserial(a, b):
    """Rank-biserial correlation as effect size for Mann-Whitney U."""
    na, nb = len(a), len(b)
    u, _ = stats.mannwhitneyu(a, b, alternative='two-sided')
    return 1 - (2 * u) / (na * nb)


def test_roi_group_differences(merged: pd.DataFrame,
                                roi_cols: list,
                                alpha: float = 0.05) -> pd.DataFrame:
    """
    Two-sided Mann-Whitney U test per ROI comparing progressor vs non-progressor
    predicted slopes. BH FDR correction across all ROIs.
    Effect size: rank-biserial correlation (r = 1 − 2U/n₁n₂).
    """
    prog     = merged[merged['is_progressor'] == 1]
    non_prog = merged[merged['is_progressor'] == 0]

    results = []
    for col in roi_cols:
        a = prog[col].dropna().values
        b = non_prog[col].dropna().values
        if len(a) < 3 or len(b) < 3:
            continue
        u_stat, p_val = stats.mannwhitneyu(a, b, alternative='two-sided')
        r_rb          = rank_biserial(a, b)
        results.append({
            'roi':          col,
            'u_stat':       u_stat,
            'p_value':      p_val,
            'effect_size':  r_rb,
            'mean_prog':    a.mean(),
            'mean_nonprog': b.mean(),
            'delta_mean':   a.mean() - b.mean(),
            'n_prog':       len(a),
            'n_nonprog':    len(b),
        })

    res_df = pd.DataFrame(results)
    if res_df.empty:
        return res_df

    _, p_adj, _, _ = multipletests(res_df['p_value'], method='fdr_bh', alpha=alpha)
    res_df['p_adjusted'] = p_adj
    res_df['significant'] = res_df['p_adjusted'] < alpha

    return res_df.sort_values('p_adjusted').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(merged: pd.DataFrame, roi_results: pd.DataFrame) -> None:
    n_prog    = int(merged['is_progressor'].sum())
    n_nonprog = int((merged['is_progressor'] == 0).sum())
    n_total   = len(roi_results)
    n_sig     = int(roi_results['significant'].sum()) if not roi_results.empty else 0

    print(f"\n{'='*72}")
    print("STAGING ANALYSIS — BrainGenFlow Trajectory Group Differences")
    print(f"{'='*72}")
    print(f"  Cohort (MCI subjects with fold predictions):")
    print(f"    Total           : {len(merged)}")
    print(f"    Progressors     : {n_prog}  ({n_prog/len(merged)*100:.1f}%)")
    print(f"    Non-progressors : {n_nonprog}  ({n_nonprog/len(merged)*100:.1f}%)")
    print(f"\n  ROI-level Mann-Whitney U test (BH FDR, α = 0.05):")
    print(f"    ROIs tested  : {n_total}")
    print(f"    Significant  : {n_sig}  ({n_sig/n_total*100:.1f}%)" if n_total else "    ROIs tested  : 0")

    if n_sig > 0:
        top = roi_results[roi_results['significant']].head(20)
        print(f"\n  Top significant ROIs (up to 20) — sorted by adjusted p-value:")
        print(f"  {'ROI':<12} {'p_adj':>10} {'effect_r':>10} {'Δmean':>9} "
              f"{'mean_prog':>11} {'mean_nonprog':>13}")
        print("  " + "-" * 70)
        for _, row in top.iterrows():
            print(f"  {row['roi']:<12} {row['p_adjusted']:>10.3e} "
                  f"{row['effect_size']:>10.3f} {row['delta_mean']:>9.5f} "
                  f"{row['mean_prog']:>11.5f} {row['mean_nonprog']:>13.5f}")
    else:
        print("\n  No ROIs survive FDR correction.")

    frac = n_sig / n_total if n_total > 0 else 0
    if   frac >= 0.10: verdict = "MAINTAINED  — predicted trajectories preserve substantial group differences."
    elif frac >= 0.02: verdict = "PARTIAL     — weak but present group separation."
    else:              verdict = "NOT MAINTAINED — predicted trajectories lose group differences."

    print(f"\n  Overall verdict: {verdict}")
    print(f"{'='*72}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / 'staging_analysis_results.txt'

    class Tee:
        def __init__(self, path):
            self._file   = open(path, 'w')
            self._stdout = sys.stdout
        def write(self, txt):
            self._stdout.write(txt)
            self._file.write(txt)
        def flush(self):
            self._stdout.flush()
            self._file.flush()
        def close(self):
            self._file.close()

    tee = Tee(log_path)
    sys.stdout = tee

    try:
        print("BrainGenFlow Staging Analysis")
        print(f"Working directory: {Path.cwd()}")
        print()

        # ── 1. ROI labels ────────────────────────────────────────────────────
        print("Step 1: Building ROI label mapping …")
        roi_labels = build_roi_labels(HMUSE_FILE)

        # ── 2. Predicted slopes from _gen trajectories ───────────────────────
        print("\nStep 2: Loading fold trajectories and computing OLS slopes …")
        slopes_df = load_predicted_slopes(PRED_DIR, roi_labels)

        roi_cols = [c for c in slopes_df.columns if c != 'PTID']
        print(f"  ROI columns in slopes DataFrame: "
              f"{roi_cols[:4]} … {roi_cols[-3:]}")

        # ── 3. Progression labels ────────────────────────────────────────────
        print("\nStep 3: Building MCI progressor / non-progressor labels …")
        prog_df = build_progression_labels(COV_FILE)
        print(f"  MCI subjects in covariates : {len(prog_df)}")
        print(f"  Progressors                : {prog_df['is_progressor'].sum()}")
        print(f"  Non-progressors            : {(prog_df['is_progressor']==0).sum()}")

        # ── 4. Merge ─────────────────────────────────────────────────────────
        print("\nStep 4: Merging slopes with progression labels …")
        merged = pd.merge(slopes_df, prog_df, on='PTID', how='inner')
        merged = merged.sort_values('PTID').reset_index(drop=True)

        # Identify the specific ROI columns present in the merged DataFrame
        roi_cols_merged = [c for c in merged.columns
                           if c.startswith('MUSE_') or c.startswith('ROI_')]
        print(f"  Subjects after merge         : {len(merged)}")
        print(f"  ROI columns in merged df     : {len(roi_cols_merged)}")
        print(f"  Specific ROI cols (first 5)  : {roi_cols_merged[:5]}")
        print(f"  Specific ROI cols (last  5)  : {roi_cols_merged[-5:]}")

        n_before = len(merged)
        merged = merged.dropna(subset=roi_cols_merged, how='all').reset_index(drop=True)
        print(f"  Subjects after all-NaN drop  : {len(merged)}  "
              f"(removed {n_before - len(merged)})")

        # ── 5. Statistical tests ─────────────────────────────────────────────
        print("\nStep 5: Mann-Whitney U test per ROI (BH FDR correction) …")
        roi_results = test_roi_group_differences(merged, roi_cols_merged)

        # ── 6. Report & save ─────────────────────────────────────────────────
        print_summary(merged, roi_results)

        merged_out  = OUT_DIR / 'merged_mci_slopes_with_labels.csv'
        roi_res_out = OUT_DIR / 'roi_group_difference_tests.csv'
        merged.to_csv(merged_out, index=False)
        roi_results.to_csv(roi_res_out, index=False)
        print(f"  Saved merged DataFrame  → {merged_out}")
        print(f"  Saved ROI test results  → {roi_res_out}")

        return merged, roi_results

    finally:
        sys.stdout = tee._stdout
        tee.close()
        print(f"Full log saved to: {log_path}")


if __name__ == '__main__':
    main()
