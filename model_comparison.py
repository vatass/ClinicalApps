"""
NeurIPS 2026: Model Comparison Table
=====================================
Computes per-fold MAE and mean MAE ± 95 % CI for three methods:
  - DKGP   (our method)  — from predictions/trajectory_<PTID>_fold<N>.csv
  - MLP    (baseline)    — from baselines/mlp/fold_<N>/
  - RNN-AD (baseline)    — from baselines/rnn/fold_<N>/

Also saves:
  - results/mlp_predictions.csv   : PTID, Time, Fold, ROI_0 … ROI_144 (predicted + actual)
  - results/rnn_predictions.csv   : same format
"""

import json
import os
import re
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
PRED_DIR = "./predictions"
MLP_DIR  = "./baselines/mlp"
RNN_DIR  = "./baselines/rnn"
COV_FILE = "./longitudinal_covariates_allstudies.csv"
OUT_DIR  = "./results"
N_FOLDS  = 5
N_ROIS   = 145

os.makedirs(OUT_DIR, exist_ok=True)


# ── Statistics helpers ────────────────────────────────────────────────────────

def mean_ci(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Return (mean, ci_lower, ci_upper) using the t-distribution."""
    n = len(values)
    m = float(np.mean(values))
    if n < 2:
        return m, m, m
    se     = stats.sem(values)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    h      = t_crit * se
    return m, m - h, m + h


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DKGP — parse trajectory CSVs grouped by fold
# ─────────────────────────────────────────────────────────────────────────────

def _parse_fold_file(path: str) -> Tuple[str, int, pd.DataFrame]:
    """
    Parse one trajectory_<PTID>_fold<N>.csv.
    Returns (ptid, fold, obs_df) where obs_df has columns:
        ROI_Index, Month, Real, Predicted
    Only rows where Real is non-NaN are kept.
    Month 0 is excluded (prediction equals observation by construction).
    """
    fname = os.path.basename(path)
    m     = re.match(r"trajectory_(.+)_fold(\d+)\.csv$", fname)
    if m is None:
        return None, None, None
    ptid = m.group(1)
    fold = int(m.group(2))

    df = pd.read_csv(path)
    real_rows = df[df["ROI"].str.endswith("_real")].copy()
    gen_rows  = df[df["ROI"].str.endswith("_gen")].copy()

    real_rows["ROI_Index"] = real_rows["ROI"].str.replace("_real", "").astype(int)
    gen_rows["ROI_Index"]  = gen_rows["ROI"].str.replace("_gen",  "").astype(int)

    month_cols = [c for c in df.columns if re.match(r"^Month_\d+$", c)]

    records = []
    for _, rrow in real_rows.iterrows():
        roi_idx  = rrow["ROI_Index"]
        gen_row  = gen_rows[gen_rows["ROI_Index"] == roi_idx]
        if gen_row.empty:
            continue
        gen_row = gen_row.iloc[0]

        for col in month_cols:
            real_val = rrow[col]
            if pd.isna(real_val):
                continue
            month = int(col.split("_")[1])
            if month == 0:          # T=0: model is initialised on actual value
                continue
            records.append({
                "ROI_Index": roi_idx,
                "Month":     month,
                "Real":      real_val,
                "Predicted": gen_row[col],
            })

    return ptid, fold, pd.DataFrame(records)


def compute_dkgp_fold_maes() -> Dict[int, float]:
    """
    Return {fold: MAE} for DKGP across all trajectory CSV files.
    MAE = mean |Predicted − Real| over all (ROI, timepoint) pairs in the fold.
    """
    files = sorted(
        f for f in os.listdir(PRED_DIR)
        if re.match(r"trajectory_.+_fold\d+\.csv$", f)
    )
    print(f"[DKGP] Found {len(files)} fold trajectory files.")

    fold_errors: Dict[int, List[float]] = {k: [] for k in range(N_FOLDS)}

    for i, fname in enumerate(files):
        if i % 500 == 0:
            print(f"  [{i}/{len(files)}] …")
        ptid, fold, obs = _parse_fold_file(os.path.join(PRED_DIR, fname))
        if ptid is None or obs.empty:
            continue
        abs_errors = np.abs(obs["Predicted"].values - obs["Real"].values)
        fold_errors[fold].extend(abs_errors.tolist())

    fold_maes = {}
    for fold, errs in fold_errors.items():
        if errs:
            fold_maes[fold] = float(np.mean(errs))
    return fold_maes


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MLP — load numpy arrays per fold
# ─────────────────────────────────────────────────────────────────────────────

def compute_mlp_fold_maes() -> Dict[int, float]:
    """Return {fold: MAE} for MLP. MAE averaged over all observations × 145 ROIs."""
    fold_maes = {}
    for fold in range(N_FOLDS):
        preds = np.load(os.path.join(MLP_DIR, f"fold_{fold}", "test_predictions.npy"))
        tgts  = np.load(os.path.join(MLP_DIR, f"fold_{fold}", "test_targets.npy"))
        fold_maes[fold] = float(np.mean(np.abs(preds - tgts)))
        print(f"[MLP]  fold {fold}: MAE={fold_maes[fold]:.4f}  "
              f"n_obs={preds.shape[0]}")
    return fold_maes


# ─────────────────────────────────────────────────────────────────────────────
# 3.  RNN-AD — load numpy arrays per fold
# ─────────────────────────────────────────────────────────────────────────────

def compute_rnn_fold_maes() -> Dict[int, float]:
    """Return {fold: MAE} for RNN-AD."""
    fold_maes = {}
    for fold in range(N_FOLDS):
        preds = np.load(os.path.join(RNN_DIR, f"fold_{fold}", "test_predictions.npy"))
        tgts  = np.load(os.path.join(RNN_DIR, f"fold_{fold}", "test_targets.npy"))
        fold_maes[fold] = float(np.mean(np.abs(preds - tgts)))
        print(f"[RNN]  fold {fold}: MAE={fold_maes[fold]:.4f}  "
              f"n_obs={preds.shape[0]}")
    return fold_maes


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Comparison table
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_table(
    dkgp_maes: Dict[int, float],
    mlp_maes:  Dict[int, float],
    rnn_maes:  Dict[int, float],
) -> pd.DataFrame:
    """
    Build and print the comparison table.
    Rows: DKGP, MLP, RNN-AD.
    Columns: Fold 0-4, Mean MAE, 95 % CI lower/upper.
    """
    rows = []
    for method, fold_dict in [("DKGP (ours)", dkgp_maes),
                               ("MLP",         mlp_maes),
                               ("RNN-AD",      rnn_maes)]:
        fold_vals = np.array([fold_dict[k] for k in range(N_FOLDS)])
        mean, lo, hi = mean_ci(fold_vals)
        row = {"Method": method}
        for k in range(N_FOLDS):
            row[f"Fold {k}"] = round(fold_vals[k], 4)
        row["Mean MAE"]  = round(mean, 4)
        row["CI_lower"]  = round(lo,   4)
        row["CI_upper"]  = round(hi,   4)
        row["95 % CI"]   = f"[{lo:.4f}, {hi:.4f}]"
        rows.append(row)

    tbl = pd.DataFrame(rows)

    # ── Print formatted table ──────────────────────────────────────────────
    width = 80
    print("\n" + "=" * width)
    print("MODEL COMPARISON — MEAN ABSOLUTE ERROR (MAE)")
    print("  Metric: mean |predicted − real| over all (observation × ROI) pairs")
    print("  95 % CI: t-distribution, df = 4  (5-fold cross-validation)")
    print("  DKGP excludes Month 0 (model initialised on actual baseline value)")
    print("=" * width)

    col_w = 9
    header = (f"  {'Method':<18}"
              + "".join(f"  {'Fold ' + str(k):>{col_w}}" for k in range(N_FOLDS))
              + f"  {'Mean MAE':>{col_w}}  {'95 % CI'}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for _, r in tbl.iterrows():
        fold_vals_str = "".join(
            f"  {r[f'Fold {k}']:>{col_w}.4f}" for k in range(N_FOLDS)
        )
        # Bold best (lowest) MAE per column using ▸ marker
        print(f"  {r['Method']:<18}{fold_vals_str}"
              f"  {r['Mean MAE']:>{col_w}.4f}  {r['95 % CI']}")

    print("=" * width)

    # Mark best (lowest mean MAE) method
    best_idx = tbl["Mean MAE"].idxmin()
    print(f"\n  Best method: {tbl.loc[best_idx, 'Method']}  "
          f"(mean MAE = {tbl.loc[best_idx, 'Mean MAE']:.4f})\n")

    # Per-fold best
    for k in range(N_FOLDS):
        col   = f"Fold {k}"
        best  = tbl.loc[tbl[col].idxmin(), "Method"]
        worst = tbl.loc[tbl[col].idxmax(), "Method"]
        print(f"  Fold {k}: best={best}  worst={worst}")

    print()

    # Save
    out_path = os.path.join(OUT_DIR, "model_comparison_table.csv")
    tbl.to_csv(out_path, index=False)
    print(f"[table] Saved: {out_path}")
    return tbl


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Save MLP and RNN prediction files (PTID, Time, Fold, ROI_0…ROI_144)
# ─────────────────────────────────────────────────────────────────────────────

def _get_ptid_time_map(cov_file: str) -> Dict[str, List[int]]:
    """
    For each PTID, return the sorted list of visit Times from the covariate file.
    Used to map visit index → actual month.
    """
    cov = pd.read_csv(cov_file, usecols=["PTID", "Time"])
    return (cov.sort_values("Time")
               .groupby("PTID")["Time"]
               .apply(list)
               .to_dict())


def save_baseline_predictions(model: str) -> None:
    """
    Concatenate test_predictions.npy and test_targets.npy across all folds
    into one CSV with columns:
        PTID, Time, Fold,
        pred_ROI_0 … pred_ROI_144,
        real_ROI_0 … real_ROI_144
    Time is looked up from the covariate file by matching visit order per subject.
    """
    assert model in ("mlp", "rnn")
    base_dir   = MLP_DIR if model == "mlp" else RNN_DIR
    roi_cols_p = [f"pred_ROI_{i}" for i in range(N_ROIS)]
    roi_cols_r = [f"real_ROI_{i}" for i in range(N_ROIS)]

    ptid_time_map = _get_ptid_time_map(COV_FILE)

    all_frames = []
    for fold in range(N_FOLDS):
        fold_dir = os.path.join(base_dir, f"fold_{fold}")
        preds    = np.load(os.path.join(fold_dir, "test_predictions.npy"))
        tgts     = np.load(os.path.join(fold_dir, "test_targets.npy"))

        # Get PTID for each row
        ptid_file = os.path.join(fold_dir, "test_ptids.json")
        if not os.path.exists(ptid_file):
            # RNN uses subject_ids.json — rows are ordered by (subject, visit)
            with open(os.path.join(fold_dir, "subject_ids.json")) as f:
                ids = json.load(f)
            test_subjects = ids["test"]
            # Expand: repeat each subject ID for each of its visits
            ptid_list = []
            for subj in test_subjects:
                n_visits = len(ptid_time_map.get(str(subj), []))
                ptid_list.extend([str(subj)] * max(n_visits, 1))
            # Trim or pad to match actual array length.
            # The covariate file may not cover every visit in the RNN predictions
            # (e.g. subjects with extra timepoints not in the covariate CSV).
            if len(ptid_list) > len(preds):
                ptid_list = ptid_list[:len(preds)]
            elif len(ptid_list) < len(preds):
                n_missing = len(preds) - len(ptid_list)
                print(f"  [RNN] fold {fold}: covariate file has {len(ptid_list)} rows "
                      f"but predictions array has {len(preds)} — padding {n_missing} rows with NaN.")
                ptid_list = ptid_list + [""] * n_missing
        else:
            with open(ptid_file) as f:
                ptid_list = json.load(f)

        # Map each row to the corresponding Time using visit-order counter per PTID
        visit_counter: Dict[str, int] = {}
        times = []
        for ptid in ptid_list:
            idx   = visit_counter.get(ptid, 0)
            visits = ptid_time_map.get(str(ptid), [])
            t      = visits[idx] if idx < len(visits) else np.nan
            times.append(t)
            visit_counter[ptid] = idx + 1

        frame = pd.DataFrame(
            preds, columns=roi_cols_p
        )
        frame_real = pd.DataFrame(
            tgts, columns=roi_cols_r
        )
        frame = pd.concat([frame, frame_real], axis=1)
        frame.insert(0, "Fold",  fold)
        frame.insert(0, "Time",  times)
        frame.insert(0, "PTID",  ptid_list)
        all_frames.append(frame)

    merged = pd.concat(all_frames, ignore_index=True)
    out_path = os.path.join(OUT_DIR, f"{model}_predictions.csv")
    merged.to_csv(out_path, index=False)
    print(f"[{model.upper():6s}] Saved {len(merged):,} rows × "
          f"{len(merged.columns)} cols → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("STEP 1 — DKGP per-fold MAE  (from trajectory CSV files)")
    print("=" * 70)
    dkgp_maes = compute_dkgp_fold_maes()
    print("  DKGP fold MAEs:", {k: round(v, 4) for k, v in dkgp_maes.items()})

    print("\n" + "=" * 70)
    print("STEP 2 — MLP per-fold MAE")
    print("=" * 70)
    mlp_maes = compute_mlp_fold_maes()

    print("\n" + "=" * 70)
    print("STEP 3 — RNN-AD per-fold MAE")
    print("=" * 70)
    rnn_maes = compute_rnn_fold_maes()

    print("\n" + "=" * 70)
    print("STEP 4 — Comparison table")
    print("=" * 70)
    build_comparison_table(dkgp_maes, mlp_maes, rnn_maes)

    print("=" * 70)
    print("STEP 5 — Saving MLP prediction file")
    print("=" * 70)
    save_baseline_predictions("mlp")

    print("\n" + "=" * 70)
    print("STEP 6 — Saving RNN-AD prediction file")
    print("=" * 70)
    save_baseline_predictions("rnn")

    print("\n" + "=" * 70)
    print("DONE — all outputs in", OUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
