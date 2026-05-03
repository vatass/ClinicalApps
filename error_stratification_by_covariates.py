'''
NeuRIPS 2026: Error Stratification by Covariates

Evaluates whether BrainGenFlow predicted trajectories maintain group differences
between MCI progressors and non-progressors. For each subject, an OLS slope is
computed per ROI across the predicted trajectory, then a wide feature matrix
(subjects × ROIs) is merged with progression labels and tested ROI-by-ROI.
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
PRED_DIR  = Path('./predictions')
COV_FILE  = Path('./longitudinal_covariates_allstudies.csv')
HMUSE_FILE = Path('./hmuse_list.npy')   # list-index → MUSE ROI number mapping
OUT_DIR   = Path('./staging_analysis')

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
# ROI label helper
# ---------------------------------------------------------------------------
def build_roi_labels(n_rois: int, hmuse_path: Path) -> dict:
    """
    Return a mapping from list index (0-based) to ROI label string.

    If hmuse_list.npy exists, each label is 'MUSE_<actual_roi_number>';
    otherwise labels fall back to 'ROI_<list_index>'.
    """
    if hmuse_path.exists():
        hmuse = np.load(str(hmuse_path), allow_pickle=True)
        print(f"  Loaded hmuse_list.npy  → {len(hmuse)} entries")
        return {i: f'MUSE_{int(hmuse[i])}' for i in range(min(n_rois, len(hmuse)))}
    else:
        print(f"  hmuse_list.npy not found — using list-index labels (ROI_0 … ROI_{n_rois-1})")
        return {i: f'ROI_{i}' for i in range(n_rois)}


# ---------------------------------------------------------------------------
# Load BrainGenFlow trajectories & compute OLS slopes
# ---------------------------------------------------------------------------
def load_predicted_slopes(pred_dir: Path, roi_labels: dict) -> pd.DataFrame:
    """
    For every subject trajectory CSV in pred_dir, compute one OLS slope per ROI
    across the monthly predicted values. Returns a wide DataFrame:

        columns: PTID, ROI_<i> (or MUSE_<n>) × 145
    """
    csv_files = sorted(pred_dir.glob('trajectory_ptid_*.csv'))
    print(f"  Found {len(csv_files)} trajectory files")

    month_range = np.arange(120, dtype=float)   # Month_0 … Month_119
    month_cols  = [f'Month_{m}' for m in range(120)]

    rows = []
    for fpath in csv_files:
        ptid = re.sub(r'^trajectory_ptid_(.*)\.csv$', r'\1', fpath.name)
        try:
            df = pd.read_csv(fpath)
        except Exception:
            continue

        if 'ROI_Index' not in df.columns:
            continue

        roi_slopes = {'PTID': ptid}
        for _, row in df.iterrows():
            idx   = int(row['ROI_Index'])
            label = roi_labels.get(idx, f'ROI_{idx}')
            vals  = row[month_cols].values.astype(float)
            if np.isnan(vals).any():
                slope = np.nan
            else:
                slope = np.polyfit(month_range, vals, 1)[0]
            roi_slopes[label] = slope

        rows.append(roi_slopes)

    slopes_df = pd.DataFrame(rows)
    print(f"  Computed slopes for {len(slopes_df)} subjects, "
          f"{slopes_df.shape[1] - 1} ROIs")
    return slopes_df


# ---------------------------------------------------------------------------
# Build progression labels from covariates
# ---------------------------------------------------------------------------
def build_progression_labels(cov_file: Path) -> pd.DataFrame:
    """
    Returns a DataFrame with columns [PTID, is_progressor] for all subjects
    whose first recorded diagnosis is MCI. is_progressor=1 if any later visit
    carries an AD/Dementia diagnosis.
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
# ROI-level Mann-Whitney U tests (progressors vs non-progressors)
# ---------------------------------------------------------------------------
def test_roi_group_differences(merged: pd.DataFrame,
                                roi_cols: list,
                                alpha: float = 0.05) -> pd.DataFrame:
    """
    For each ROI, run a two-sided Mann-Whitney U test comparing slope
    distributions between progressors (is_progressor=1) and
    non-progressors (is_progressor=0). Applies Benjamini-Hochberg FDR
    correction across all ROIs.

    Returns a DataFrame sorted by adjusted p-value.
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
        mean_prog     = a.mean()
        mean_nonprog  = b.mean()
        results.append({
            'roi':          col,
            'u_stat':       u_stat,
            'p_value':      p_val,
            'mean_prog':    mean_prog,
            'mean_nonprog': mean_nonprog,
            'delta_mean':   mean_prog - mean_nonprog,
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
# Summary printing
# ---------------------------------------------------------------------------
def print_summary(merged: pd.DataFrame, roi_results: pd.DataFrame) -> None:
    n_prog    = merged['is_progressor'].sum()
    n_nonprog = (merged['is_progressor'] == 0).sum()
    n_sig     = roi_results['significant'].sum()
    n_total   = len(roi_results)

    print(f"\n{'='*70}")
    print("STAGING ANALYSIS — BrainGenFlow Trajectory Group Differences")
    print(f"{'='*70}")
    print(f"  Cohort (MCI subjects with predictions):")
    print(f"    Total        : {len(merged)}")
    print(f"    Progressors  : {n_prog}  ({n_prog/len(merged)*100:.1f}%)")
    print(f"    Non-progressors: {n_nonprog}  ({n_nonprog/len(merged)*100:.1f}%)")
    print(f"\n  ROI-level Mann-Whitney U test (BH-corrected, α=0.05):")
    print(f"    ROIs tested  : {n_total}")
    print(f"    Significant  : {n_sig}  ({n_sig/n_total*100:.1f}%)")

    if n_sig > 0:
        top = roi_results[roi_results['significant']].head(20)
        print(f"\n  Top significant ROIs (up to 20):")
        print(f"  {'ROI':<15} {'p_adj':>10} {'Δmean':>10} {'mean_prog':>12} {'mean_nonprog':>14}")
        print("  " + "-" * 65)
        for _, row in top.iterrows():
            print(f"  {row['roi']:<15} {row['p_adjusted']:>10.4e} "
                  f"{row['delta_mean']:>10.4f} "
                  f"{row['mean_prog']:>12.4f} {row['mean_nonprog']:>14.4f}")
    else:
        print("\n  No ROIs survive FDR correction — predicted trajectories may not")
        print("  preserve the progressor/non-progressor separation at α=0.05.")

    print(f"\n  Overall verdict:")
    frac = n_sig / n_total if n_total > 0 else 0
    if frac >= 0.10:
        verdict = "MAINTAINED — predicted trajectories preserve substantial group differences."
    elif frac >= 0.02:
        verdict = "PARTIALLY MAINTAINED — weak but present group separation."
    else:
        verdict = "NOT MAINTAINED — predicted trajectories lose group differences."
    print(f"    {verdict}")
    print(f"{'='*70}\n")


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

        # ── 1. ROI label mapping ─────────────────────────────────────────────
        print("Step 1: Building ROI label mapping …")
        roi_labels = build_roi_labels(145, HMUSE_FILE)

        # ── 2. Load predicted slopes ─────────────────────────────────────────
        print("\nStep 2: Loading predicted trajectories and computing OLS slopes …")
        slopes_df = load_predicted_slopes(PRED_DIR, roi_labels)

        roi_cols = [c for c in slopes_df.columns if c != 'PTID']
        print(f"  ROI columns in slopes DataFrame: {roi_cols[:5]} … {roi_cols[-3:]}")

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

        # Identify the specific ROI columns present after the merge
        roi_cols_merged = [c for c in merged.columns
                           if c.startswith('ROI_') or c.startswith('MUSE_')]
        print(f"  Subjects after merge        : {len(merged)}")
        print(f"  ROI columns in merged df    : {len(roi_cols_merged)}")
        print(f"  First 5 ROI cols            : {roi_cols_merged[:5]}")
        print(f"  Last  5 ROI cols            : {roi_cols_merged[-5:]}")

        # Drop rows with all-NaN slopes
        n_before = len(merged)
        merged = merged.dropna(subset=roi_cols_merged, how='all').reset_index(drop=True)
        print(f"  Subjects after NaN drop     : {len(merged)}  "
              f"(removed {n_before - len(merged)})")

        # ── 5. ROI-level statistical tests ────────────────────────────────────
        print("\nStep 5: Running Mann-Whitney U tests per ROI (BH FDR correction) …")
        roi_results = test_roi_group_differences(merged, roi_cols_merged)

        # ── 6. Print & save results ──────────────────────────────────────────
        print_summary(merged, roi_results)

        merged_out = OUT_DIR / 'merged_mci_slopes_with_labels.csv'
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
