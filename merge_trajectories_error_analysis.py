"""
NeuRIPS 2026: Merge predicted trajectories and analyse prediction error
stratified by longitudinal covariates.

Steps
-----
1. Merge all per-subject trajectory CSVs in predictions/ into one combined CSV.
2. At each subject's observed time-point (covariate file), extract the
   predicted value and the change-from-baseline (delta) for every ROI.
3. Compute MAE / mean-delta per covariate stratum.
4. Produce publication-quality figures: error vs Time, per covariate.
"""

import os
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", message=".*seaborn styles.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
PRED_DIR   = "./predictions"
COV_FILE   = "./longitudinal_covariates_allstudies.csv"
OUT_DIR    = "./trajectory_error_analysis"
MERGED_CSV = os.path.join(OUT_DIR, "merged_trajectories.csv")
ERROR_CSV  = os.path.join(OUT_DIR, "error_by_covariates.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# ── Plot style ─────────────────────────────────────────────────────────────────
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("ggplot")

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "figure.dpi": 150,
})

PALETTE = sns.color_palette("tab10")

# ─────────────────────────────────────────────────────────────────────────────
# 1. MERGE ALL TRAJECTORY CSV FILES
# ─────────────────────────────────────────────────────────────────────────────

def merge_trajectories(pred_dir: str, out_path: str) -> pd.DataFrame:
    """
    Read every predictions/trajectory_ptid_*.csv, add a PTID column,
    and stack them into a single wide DataFrame.

    Output columns: PTID, ROI_Index, Month_0, Month_1, ..., Month_119
    """
    if os.path.exists(out_path):
        print(f"[merge] Loading cached merged file: {out_path}")
        return pd.read_csv(out_path)

    files = sorted(
        f for f in os.listdir(pred_dir)
        if f.startswith("trajectory_ptid_") and f.endswith(".csv")
    )
    print(f"[merge] Found {len(files)} trajectory files.")

    frames = []
    for fname in files:
        ptid = re.sub(r"^trajectory_ptid_|\.csv$", "", fname)
        df = pd.read_csv(os.path.join(pred_dir, fname))
        frames.append(pd.concat([pd.DataFrame({"PTID": [ptid] * len(df)}), df], axis=1))

    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(out_path, index=False)
    print(f"[merge] Saved {len(merged):,} rows → {out_path}")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 2. BUILD ERROR TABLE
# ─────────────────────────────────────────────────────────────────────────────

def _clean_diagnosis(dx: str) -> str:
    """Map heterogeneous diagnosis strings to CN / MCI / AD / Other."""
    if not isinstance(dx, str):
        return "Other"
    dx_lo = dx.strip().lower()
    if dx_lo in ("cn", "cognitively normal", "cognitively unimpaired",
                 "cn assumed by study criteria", "normal cognition", "normal",
                 "memory complainer (healthy control)"):
        return "CN"
    if "mci" in dx_lo or "mild cognitive" in dx_lo:
        return "MCI"
    if "ad" in dx_lo or "alzheimer" in dx_lo or "dementia" in dx_lo:
        return "AD"
    return "Other"


def _age_group(age: float) -> str:
    if age < 60:
        return "<60"
    if age < 70:
        return "60-70"
    if age < 80:
        return "70-80"
    return "≥80"


def build_error_table(merged: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    """
    For every (subject, observed-time-point) in the covariate file:
      - look up the predicted trajectory value at that month
      - compute change-from-baseline  delta(T) = pred(T) – pred(0)
      - compute mean |delta| across all ROIs  → per-row MAE proxy

    Returns a long DataFrame with one row per (PTID, Time).
    """
    month_cols = [c for c in merged.columns if re.match(r"^Month_\d+$", c)]
    max_month  = max(int(c.split("_")[1]) for c in month_cols)

    # Pre-compute baseline (Month_0) per (PTID, ROI_Index) for speed
    baseline = (
        merged[["PTID", "ROI_Index", "Month_0"]]
        .rename(columns={"Month_0": "baseline_val"})
    )
    merged_b = merged.merge(baseline, on=["PTID", "ROI_Index"])

    # Keep only subjects present in both files
    common = set(merged["PTID"].unique()) & set(cov["PTID"].unique())
    print(f"[error] Subjects in both files: {len(common)}")

    cov_sub = cov[cov["PTID"].isin(common)].copy()
    cov_sub["Diagnosis_clean"] = cov_sub["Diagnosis"].apply(_clean_diagnosis)
    cov_sub["Age_group"]       = cov_sub["Age"].apply(_age_group)
    cov_sub["Sex_label"]       = cov_sub["Sex"].map({0: "Female", 1: "Male"}).fillna("Unknown")
    cov_sub["APOE4_label"]     = cov_sub["APOE4_Alleles"].map(
        {0.0: "ε3/ε3", 1.0: "1 ε4 allele", 2.0: "2 ε4 alleles", -1.0: "Unknown"}
    ).fillna("Unknown")

    records = []
    traj_by_ptid = {ptid: grp for ptid, grp in merged_b.groupby("PTID")}

    for _, cov_row in cov_sub.iterrows():
        ptid = cov_row["PTID"]
        t    = int(cov_row["Time"])
        if t > max_month or ptid not in traj_by_ptid:
            continue
        month_col = f"Month_{t}"
        if month_col not in merged_b.columns:
            continue

        grp = traj_by_ptid[ptid]
        pred_vals     = grp[month_col].values
        baseline_vals = grp["baseline_val"].values
        delta         = pred_vals - baseline_vals          # per-ROI change from baseline

        records.append({
            "PTID":             ptid,
            "Time":             t,
            "mean_pred":        pred_vals.mean(),
            "mean_delta":       delta.mean(),
            "mean_abs_delta":   np.abs(delta).mean(),
            "std_delta":        delta.std(),
            "Diagnosis":        cov_row["Diagnosis_clean"],
            "Sex":              cov_row["Sex_label"],
            "APOE4":            cov_row["APOE4_label"],
            "Age_group":        cov_row["Age_group"],
            "Study":            cov_row["Study"],
            "Age":              cov_row["Age"],
            "Hypertension":     cov_row.get("Hypertension", np.nan),
            "Diabetes":         cov_row.get("Diabetes", np.nan),
        })

    error_df = pd.DataFrame(records)
    error_df.to_csv(ERROR_CSV, index=False)
    print(f"[error] Saved {len(error_df):,} rows → {ERROR_CSV}")
    return error_df


# ─────────────────────────────────────────────────────────────────────────────
# 3. VISUALISATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _bin_time(df: pd.DataFrame, bin_width: int = 12) -> pd.DataFrame:
    """Add a Time_bin column (bin centre) for smooth trend plots."""
    df = df.copy()
    bins = np.arange(0, df["Time"].max() + bin_width, bin_width)
    labels = (bins[:-1] + bins[1:]) / 2
    df["Time_bin"] = pd.cut(df["Time"], bins=bins, labels=labels, include_lowest=True)
    df["Time_bin"] = df["Time_bin"].astype(float)
    return df


def _compute_ci(group: pd.Series, confidence: float = 0.95) -> tuple[float, float]:
    """Return (mean, half-width of CI) using t-distribution."""
    n  = len(group)
    if n < 2:
        return group.mean(), 0.0
    se = stats.sem(group, nan_policy="omit")
    h  = se * stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return group.mean(), h


def plot_error_vs_time_by_covariate(
    error_df: pd.DataFrame,
    covariate: str,
    covariate_label: str,
    metric: str = "mean_abs_delta",
    metric_label: str = "Mean |ΔPrediction| from Baseline",
    order: list | None = None,
    palette: list | None = None,
    filename: str | None = None,
) -> None:
    """Line plot: error metric vs binned Time, stratified by covariate."""
    df = _bin_time(error_df.dropna(subset=[covariate, metric]))
    groups = order or sorted(df[covariate].dropna().unique())
    pal    = palette or PALETTE

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, grp_val in enumerate(groups):
        sub = df[df[covariate] == grp_val]
        agg = (
            sub.groupby("Time_bin")[metric]
            .apply(lambda x: _compute_ci(x))
            .reset_index()
        )
        agg[["mean", "ci_h"]] = pd.DataFrame(agg[metric].tolist(), index=agg.index)
        agg = agg.dropna(subset=["Time_bin"])

        color = pal[i % len(pal)]
        ax.plot(agg["Time_bin"], agg["mean"], label=str(grp_val),
                color=color, linewidth=2.2, marker="o", markersize=4)
        ax.fill_between(
            agg["Time_bin"],
            agg["mean"] - agg["ci_h"],
            agg["mean"] + agg["ci_h"],
            alpha=0.15, color=color
        )

    ax.set_xlabel("Time (months)", fontweight="bold")
    ax.set_ylabel(metric_label, fontweight="bold")
    ax.set_title(f"Prediction Error vs Time\nStratified by {covariate_label}",
                 fontweight="bold", pad=10)
    ax.legend(title=covariate_label, framealpha=0.9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(24))
    sns.despine(ax=ax)
    plt.tight_layout()

    fname = filename or os.path.join(OUT_DIR, f"error_vs_time_{covariate.lower()}.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved: {fname}")


def plot_boxplot_by_covariate(
    error_df: pd.DataFrame,
    covariate: str,
    covariate_label: str,
    metric: str = "mean_abs_delta",
    metric_label: str = "Mean |ΔPrediction| from Baseline",
    order: list | None = None,
    filename: str | None = None,
) -> None:
    """Box plot: error distribution per covariate group (all time points pooled)."""
    df = error_df.dropna(subset=[covariate, metric])
    groups = order or sorted(df[covariate].dropna().unique())

    fig, ax = plt.subplots(figsize=(max(6, len(groups) * 1.8), 5))
    sns.boxplot(
        data=df, x=covariate, y=metric,
        order=groups, palette="tab10",
        width=0.55, fliersize=2, ax=ax
    )
    ax.set_xlabel(covariate_label, fontweight="bold")
    ax.set_ylabel(metric_label, fontweight="bold")
    ax.set_title(f"Prediction Error Distribution by {covariate_label}",
                 fontweight="bold", pad=10)
    plt.xticks(rotation=20, ha="right")
    sns.despine(ax=ax)
    plt.tight_layout()

    fname = filename or os.path.join(OUT_DIR, f"error_boxplot_{covariate.lower()}.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved: {fname}")


def plot_heatmap_error_time_x_covariate(
    error_df: pd.DataFrame,
    covariate: str,
    covariate_label: str,
    metric: str = "mean_abs_delta",
    bin_width: int = 24,
    filename: str | None = None,
) -> None:
    """Heatmap: mean error in (Time-bin × covariate) cells."""
    df = _bin_time(error_df.dropna(subset=[covariate, metric]), bin_width=bin_width)
    pivot = df.pivot_table(
        values=metric, index=covariate, columns="Time_bin", aggfunc="mean"
    )
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 0.9), max(4, pivot.shape[0] * 0.7)))
    sns.heatmap(
        pivot, cmap="YlOrRd", annot=True, fmt=".3f",
        linewidths=0.4, cbar_kws={"label": metric}, ax=ax
    )
    ax.set_xlabel("Time bin centre (months)", fontweight="bold")
    ax.set_ylabel(covariate_label, fontweight="bold")
    ax.set_title(f"Mean Prediction Error · {covariate_label} × Time",
                 fontweight="bold", pad=10)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    fname = filename or os.path.join(OUT_DIR, f"error_heatmap_{covariate.lower()}.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved: {fname}")


def plot_combined_overview(error_df: pd.DataFrame) -> None:
    """
    Single 2×3 figure: error vs Time for the six main covariates side-by-side.
    """
    covariates = [
        ("Diagnosis",  "Diagnosis",
         ["CN", "MCI", "AD", "Other"]),
        ("Sex",        "Sex",
         ["Female", "Male"]),
        ("APOE4",      "APOE4 Genotype",
         ["ε3/ε3", "1 ε4 allele", "2 ε4 alleles", "Unknown"]),
        ("Age_group",  "Age Group",
         ["<60", "60-70", "70-80", "≥80"]),
        ("Study",      "Study",
         None),
    ]

    df = _bin_time(error_df, bin_width=12)
    metric      = "mean_abs_delta"
    metric_lbl  = "Mean |ΔPred| from Baseline"

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=False)
    axes_flat  = axes.flatten()

    for ax_idx, (col, lbl, order) in enumerate(covariates):
        ax     = axes_flat[ax_idx]
        groups = order or sorted(df[col].dropna().unique())
        pal    = PALETTE

        for i, grp_val in enumerate(groups):
            sub = df[df[col] == grp_val]
            agg = (
                sub.groupby("Time_bin")[metric]
                .apply(lambda x: _compute_ci(x))
                .reset_index()
            )
            agg[["mean", "ci_h"]] = pd.DataFrame(agg[metric].tolist(), index=agg.index)
            agg = agg.dropna(subset=["Time_bin"])
            color = pal[i % len(pal)]
            ax.plot(agg["Time_bin"], agg["mean"], label=str(grp_val),
                    color=color, linewidth=1.8, marker="o", markersize=3)
            ax.fill_between(
                agg["Time_bin"],
                agg["mean"] - agg["ci_h"],
                agg["mean"] + agg["ci_h"],
                alpha=0.12, color=color
            )

        ax.set_xlabel("Time (months)", fontsize=11)
        ax.set_ylabel(metric_lbl, fontsize=11)
        ax.set_title(lbl, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, title=lbl, title_fontsize=8, framealpha=0.85)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(24))
        sns.despine(ax=ax)

    # Last panel: overall distribution
    ax = axes_flat[5]
    bins = np.arange(0, error_df["Time"].max() + 12, 12)
    lbl_centres = (bins[:-1] + bins[1:]) / 2
    grp_means, grp_ci = [], []
    for b_lo, b_hi in zip(bins[:-1], bins[1:]):
        sub = error_df[(error_df["Time"] >= b_lo) & (error_df["Time"] < b_hi)][metric]
        if len(sub) < 2:
            grp_means.append(np.nan)
            grp_ci.append(0.0)
        else:
            m, h = _compute_ci(sub)
            grp_means.append(m)
            grp_ci.append(h)

    grp_means = np.array(grp_means)
    grp_ci    = np.array(grp_ci)
    ax.plot(lbl_centres, grp_means, color="#2C3E50", linewidth=2.2,
            marker="o", markersize=4, label="All subjects")
    ax.fill_between(lbl_centres,
                    grp_means - grp_ci,
                    grp_means + grp_ci,
                    alpha=0.2, color="#2C3E50")
    ax.set_xlabel("Time (months)", fontsize=11)
    ax.set_ylabel(metric_lbl, fontsize=11)
    ax.set_title("Overall (all subjects)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.85)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(24))
    sns.despine(ax=ax)

    fig.suptitle("Predicted Trajectory Error vs Time — Stratified by Covariates",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    fpath = os.path.join(OUT_DIR, "error_vs_time_combined_overview.png")
    plt.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved: {fpath}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. STRATIFICATION SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_stratification_summary(error_df: pd.DataFrame) -> None:
    """Print mean ± std of error metric per covariate group."""
    metric = "mean_abs_delta"
    print("\n" + "=" * 70)
    print("PREDICTION ERROR STRATIFICATION SUMMARY")
    print(f"Metric: {metric}  (mean |predicted change from baseline| across 145 ROIs)")
    print("=" * 70)

    covariates = [
        ("Diagnosis",  "Diagnosis"),
        ("Sex",        "Sex"),
        ("APOE4",      "APOE4 Genotype"),
        ("Age_group",  "Age Group"),
        ("Study",      "Study"),
    ]

    for col, label in covariates:
        print(f"\n── {label} ──")
        grp = (
            error_df.groupby(col)[metric]
            .agg(["mean", "std", "count"])
            .rename(columns={"mean": "Mean", "std": "Std", "count": "N"})
            .sort_values("Mean", ascending=False)
        )
        print(grp.to_string(float_format="{:.4f}".format))

    print("\n" + "=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("STEP 1 — Merging predicted trajectories")
    print("=" * 70)
    merged = merge_trajectories(PRED_DIR, MERGED_CSV)

    print("\n" + "=" * 70)
    print("STEP 2 — Loading covariates")
    print("=" * 70)
    cov = pd.read_csv(COV_FILE)
    print(f"  Covariates: {len(cov):,} rows, {cov['PTID'].nunique()} subjects")

    print("\n" + "=" * 70)
    print("STEP 3 — Building error table")
    print("=" * 70)
    error_df = build_error_table(merged, cov)

    # Stratification summary
    print_stratification_summary(error_df)

    print("\n" + "=" * 70)
    print("STEP 4 — Generating plots")
    print("=" * 70)

    # ── Combined overview ─────────────────────────────────────────────────────
    plot_combined_overview(error_df)

    # ── Per-covariate line plots (error vs Time) ──────────────────────────────
    plot_error_vs_time_by_covariate(
        error_df, "Diagnosis", "Diagnosis",
        order=["CN", "MCI", "AD", "Other"],
    )
    plot_error_vs_time_by_covariate(
        error_df, "Sex", "Sex",
        order=["Female", "Male"],
    )
    plot_error_vs_time_by_covariate(
        error_df, "APOE4", "APOE4 Genotype",
        order=["ε3/ε3", "1 ε4 allele", "2 ε4 alleles", "Unknown"],
    )
    plot_error_vs_time_by_covariate(
        error_df, "Age_group", "Age Group",
        order=["<60", "60-70", "70-80", "≥80"],
    )
    plot_error_vs_time_by_covariate(
        error_df, "Study", "Study",
    )

    # ── Box plots (distribution pooled across time) ───────────────────────────
    for col, label, order in [
        ("Diagnosis",  "Diagnosis",     ["CN", "MCI", "AD", "Other"]),
        ("Sex",        "Sex",           ["Female", "Male"]),
        ("APOE4",      "APOE4 Genotype", ["ε3/ε3", "1 ε4 allele", "2 ε4 alleles", "Unknown"]),
        ("Age_group",  "Age Group",     ["<60", "60-70", "70-80", "≥80"]),
        ("Study",      "Study",         None),
    ]:
        plot_boxplot_by_covariate(error_df, col, label, order=order)

    # ── Heatmaps ─────────────────────────────────────────────────────────────
    for col, label in [
        ("Diagnosis",  "Diagnosis"),
        ("Age_group",  "Age Group"),
        ("Study",      "Study"),
    ]:
        plot_heatmap_error_time_x_covariate(error_df, col, label)

    print("\n" + "=" * 70)
    print("DONE — all outputs in", OUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
