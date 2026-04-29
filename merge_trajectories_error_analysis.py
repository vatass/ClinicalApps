"""
NeurIPS 2026: Predicted trajectory error analysis stratified by covariates.

File format (new fold files)
-----------------------------
Filename : trajectory_<PTID>_fold<N>.csv
Columns  : ROI, Month_0 … Month_69
Rows     : 290 = 145 ROIs × 2 types, alternating:
             <roi_idx>_real  – sparse, actual observed values (NaN elsewhere)
             <roi_idx>_gen   – dense, model-predicted trajectory every month

Error definition
----------------
At every observed follow-up month T (Month_T where real is non-NaN, T > 0):
    error(T)  = predicted(T) − real(T)       per ROI
    MAE(T)    = mean |error(T)|              across 145 ROIs
    RMSE(T)   = sqrt(mean error(T)²)         across 145 ROIs
    Bias(T)   = mean error(T)                across 145 ROIs  (signed)

Month_0 is excluded: predicted == real by construction (shared baseline).

Outputs
-------
1. merged_observations.csv  – long table: PTID, Fold, ROI, Month, Real, Predicted, Error
2. error_by_covariates.csv  – per (PTID, Month): MAE, RMSE, Bias + all covariates
3. mae_ci_summary.csv       – mean MAE ± 95 % CI, overall and per stratum
4. NeurIPS figures (PDF + PNG, 300 dpi)
"""

import os
import re
import warnings
from typing import Optional, List

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
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ── Paths ─────────────────────────────────────────────────────────────────────
PRED_DIR    = "./predictions"
COV_FILE    = "./longitudinal_covariates_allstudies.csv"
OUT_DIR     = "./trajectory_error_analysis"
MERGED_CSV  = os.path.join(OUT_DIR, "merged_observations.csv")
ERROR_CSV   = os.path.join(OUT_DIR, "error_by_covariates.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# ── NeurIPS plot style ────────────────────────────────────────────────────────
SINGLE_COL = 3.25   # inches, single-column NeurIPS
DOUBLE_COL = 6.75   # inches, double-column NeurIPS
FIG_HEIGHT = 2.6

plt.rcParams.update({
    "font.family":           "sans-serif",
    "font.sans-serif":       ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size":             9,
    "axes.labelsize":        9,
    "axes.titlesize":        9,
    "xtick.labelsize":       8,
    "ytick.labelsize":       8,
    "legend.fontsize":       7.5,
    "legend.title_fontsize": 7.5,
    "axes.linewidth":        0.8,
    "xtick.major.width":     0.8,
    "ytick.major.width":     0.8,
    "xtick.major.size":      3,
    "ytick.major.size":      3,
    "lines.linewidth":       1.5,
    "figure.dpi":            300,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
    "savefig.pad_inches":    0.02,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.grid":             True,
    "grid.alpha":            0.3,
    "grid.linewidth":        0.5,
})

# Wong (2011) colorblind-safe palette
WONG = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
        "#E69F00", "#56B4E9", "#000000"]

METRIC     = "MAE"
METRIC_LBL = "MAE  |predicted − real|  (a.u.)"


# ─────────────────────────────────────────────────────────────────────────────
# 1. PARSE & MERGE FOLD FILES
# ─────────────────────────────────────────────────────────────────────────────

def parse_fold_filename(fname: str):
    """
    Extract (PTID, fold) from filenames like trajectory_<PTID>_fold<N>.csv.
    Returns (None, None) if the filename does not match the fold pattern.
    """
    m = re.match(r"^trajectory_(.+)_fold(\d+)\.csv$", fname)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def merge_fold_files(pred_dir: str, out_path: str) -> pd.DataFrame:
    """
    Read every trajectory_<PTID>_fold<N>.csv in pred_dir.
    For each file, separate _real and _gen rows, find months where
    real is observed (non-NaN, T > 0), and record:

        PTID, Fold, ROI, Month, Real, Predicted, Error

    Error = Predicted − Real  (signed, per ROI per observed month).
    Month_0 is excluded (error = 0 by construction).

    Saves to out_path and returns the DataFrame.
    """
    if os.path.exists(out_path):
        print(f"[merge] Loading cached file: {out_path}")
        return pd.read_csv(out_path)

    files = sorted(
        f for f in os.listdir(pred_dir)
        if re.match(r"^trajectory_.+_fold\d+\.csv$", f)
    )
    print(f"[merge] Found {len(files)} fold trajectory files.")

    month_re = re.compile(r"^Month_(\d+)$")
    records  = []

    for fname in files:
        ptid, fold = parse_fold_filename(fname)
        df = pd.read_csv(os.path.join(pred_dir, fname))

        month_cols = [c for c in df.columns if month_re.match(c)]

        real_rows = df[df["ROI"].str.endswith("_real")].copy()
        gen_rows  = df[df["ROI"].str.endswith("_gen")].copy()

        real_rows["roi_idx"] = real_rows["ROI"].str.replace("_real", "", regex=False).astype(int)
        gen_rows["roi_idx"]  = gen_rows["ROI"].str.replace("_gen",  "", regex=False).astype(int)

        real_rows = real_rows.set_index("roi_idx")
        gen_rows  = gen_rows.set_index("roi_idx")

        for col in month_cols:
            month = int(month_re.match(col).group(1))
            if month == 0:
                continue   # skip baseline — error is 0 by construction

            # Only keep ROIs where the real value is observed at this month
            real_vals = real_rows[col].dropna()
            if real_vals.empty:
                continue

            for roi_idx, real_val in real_vals.items():
                if roi_idx not in gen_rows.index:
                    continue
                pred_val = gen_rows.loc[roi_idx, col]
                records.append({
                    "PTID":      ptid,
                    "Fold":      fold,
                    "ROI":       roi_idx,
                    "Month":     month,
                    "Real":      real_val,
                    "Predicted": pred_val,
                    "Error":     pred_val - real_val,
                })

    merged = pd.DataFrame(records)
    merged.to_csv(out_path, index=False)
    print(f"[merge] Saved {len(merged):,} rows → {out_path}")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 2. BUILD ERROR TABLE  (per subject × observed month)
# ─────────────────────────────────────────────────────────────────────────────

def _clean_diagnosis(dx) -> str:
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
    if age < 60:  return "<60"
    if age < 70:  return "60–70"
    if age < 80:  return "70–80"
    return "≥80"


def build_error_table(merged: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate merged (PTID, Fold, ROI, Month, Error) to one row per
    (PTID, Month) by computing MAE, RMSE, and Bias across all 145 ROIs,
    then join with covariate metadata.
    """
    # Aggregate across ROIs → per (PTID, Month) metrics
    grp = merged.groupby(["PTID", "Fold", "Month"])["Error"]
    agg = pd.DataFrame({
        "MAE":  grp.apply(lambda x: np.abs(x).mean()),
        "RMSE": grp.apply(lambda x: np.sqrt((x**2).mean())),
        "Bias": grp.mean(),
        "N_ROI": grp.count(),
    }).reset_index()

    # ── Covariate metadata ────────────────────────────────────────────────────
    # Use the first visit per subject (baseline covariates)
    cov_base = (
        cov.sort_values("Time")
        .groupby("PTID")
        .first()
        .reset_index()
    )
    cov_base["Diagnosis"] = cov_base["Diagnosis"].apply(_clean_diagnosis)
    cov_base["Age_group"] = cov_base["Age"].apply(_age_group)
    cov_base["Sex"]       = cov_base["Sex"].map({0: "Female", 1: "Male"}).fillna("Unknown")
    cov_base["APOE4"]     = cov_base["APOE4_Alleles"].map(
        {0.0: "ε3/ε3", 1.0: "1 ε4 allele", 2.0: "2 ε4 alleles", -1.0: "Unknown"}
    ).fillna("Unknown")

    keep_cols = ["PTID", "Diagnosis", "Age_group", "Sex", "APOE4", "Study", "Age"]
    error_df  = agg.merge(cov_base[keep_cols], on="PTID", how="left")

    error_df.to_csv(ERROR_CSV, index=False)
    n_subj = error_df["PTID"].nunique()
    print(f"[error] {len(error_df):,} observations, {n_subj} subjects → {ERROR_CSV}")
    return error_df


# ─────────────────────────────────────────────────────────────────────────────
# 3. MAE + 95 % CI REPORT
# ─────────────────────────────────────────────────────────────────────────────

def _mean_ci(values: np.ndarray, confidence: float = 0.95):
    """Return (mean, ci_lower, ci_upper) using the t-distribution."""
    values = values[~np.isnan(values)]
    n = len(values)
    m = values.mean()
    if n < 2:
        return m, m, m
    h = stats.t.ppf((1 + confidence) / 2, df=n - 1) * stats.sem(values)
    return m, m - h, m + h


def report_mae_ci(error_df: pd.DataFrame) -> pd.DataFrame:
    """
    Print and save mean MAE ± 95 % CI overall and stratified by
    Diagnosis, Sex, APOE4, Age group, Study, and Fold.
    """
    rows = []

    def _add(stratum, group, values):
        m, lo, hi = _mean_ci(values)
        rows.append({"Stratum": stratum, "Group": group,
                     "N": len(values), "Mean_MAE": m,
                     "CI_lower": lo, "CI_upper": hi})

    metric = error_df[METRIC].dropna().values
    _add("Overall", "—", metric)

    covariates = [
        ("Diagnosis", ["CN", "MCI", "AD", "Other"]),
        ("Sex",       ["Female", "Male"]),
        ("APOE4",     ["ε3/ε3", "1 ε4 allele", "2 ε4 alleles", "Unknown"]),
        ("Age_group", ["<60", "60–70", "70–80", "≥80"]),
        ("Study",     None),
        ("Fold",      [0, 1, 2, 3, 4]),
    ]

    for col, order in covariates:
        groups = order or sorted(error_df[col].dropna().unique())
        for grp in groups:
            vals = error_df[error_df[col] == grp][METRIC].dropna().values
            if len(vals):
                _add(col, grp, vals)

    tbl = pd.DataFrame(rows)

    W = 74
    print("\n" + "=" * W)
    print("PREDICTION ERROR — MEAN MAE ± 95 % CI  (t-distribution)")
    print(f"  Metric : MAE = mean |predicted − real| across 145 ROIs")
    print(f"           computed at every observed follow-up month (T > 0)")
    print("=" * W)
    print(f"  {'Stratum':<12} {'Group':<22} {'N':>6}  {'Mean MAE':>10}  {'95 % CI':>22}")
    print("  " + "─" * (W - 2))

    prev = None
    for _, r in tbl.iterrows():
        if r["Stratum"] != prev:
            if prev is not None:
                print()
            prev = r["Stratum"]
        print(f"  {r['Stratum']:<12} {str(r['Group']):<22} {int(r['N']):>6}  "
              f"{r['Mean_MAE']:>10.4f}  [{r['CI_lower']:.4f}, {r['CI_upper']:.4f}]")
    print("=" * W)

    out = os.path.join(OUT_DIR, "mae_ci_summary.csv")
    tbl.to_csv(out, index=False)
    print(f"[report] Saved: {out}\n")
    return tbl


# ─────────────────────────────────────────────────────────────────────────────
# 4. VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def _agg_time(df: pd.DataFrame, group_col: Optional[str] = None,
              group_val=None, metric: str = METRIC) -> pd.DataFrame:
    """
    Aggregate metric by Month → (mean, 95 % CI half-width).
    Optionally filter to a single covariate group first.
    """
    sub = df if group_col is None else df[df[group_col] == group_val]
    sub = sub.dropna(subset=[metric])

    def _ci(x):
        m, lo, hi = _mean_ci(x.values)
        return pd.Series({"mean": m, "ci_h": (hi - lo) / 2})

    return (sub.groupby("Month")[metric]
               .apply(_ci)
               .reset_index()
               .rename(columns={"level_1": "stat"})
               .pivot(index="Month", columns="stat", values=metric)
               .reset_index())


def _save(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "png"):
        p = os.path.join(OUT_DIR, f"{stem}.{ext}")
        fig.savefig(p)
        print(f"[plot] Saved: {p}")
    plt.close(fig)


# ── helpers for consistent axes ───────────────────────────────────────────────

def _observed_months(error_df: pd.DataFrame) -> list:
    """Sorted list of months actually present in the error table."""
    return sorted(error_df["Month"].dropna().unique().astype(int).tolist())


def _set_month_ticks(ax, months: list) -> None:
    """Place x-ticks only at observed months; label every other one if crowded."""
    ax.set_xticks(months)
    labels = [str(m) if i % 2 == 0 else "" for i, m in enumerate(months)]
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlim(months[0] - 1, months[-1] + 1)


def _panel_label(ax, letter: str) -> None:
    """Add bold panel letter (a), (b) … in the top-left corner."""
    ax.text(-0.18, 1.04, f"({letter})", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left")


# ── Figure 1: combined 2×3 overview ──────────────────────────────────────────

def plot_combined_overview(error_df: pd.DataFrame) -> None:
    """
    2×3 panel figure with shared y-axis scale across all panels for direct
    comparison. Panel labels (a)–(f). No redundant y-axis labels on right
    column panels.
    """
    panels = [
        ("Diagnosis", "Diagnosis",  ["CN", "MCI", "AD", "Other"]),
        ("Sex",       "Sex",        ["Female", "Male"]),
        ("APOE4",     "APOE4",      ["ε3/ε3", "1 ε4 allele", "2 ε4 alleles", "Unknown"]),
        ("Age_group", "Age group",  ["<60", "60–70", "70–80", "≥80"]),
        ("Study",     "Study",      None),
    ]
    letters  = "abcdef"
    months   = _observed_months(error_df)

    # Compute global y-limits across all groups for shared scale
    all_means, all_ci = [], []
    for col, _, order in panels:
        groups = order or sorted(error_df[col].dropna().unique())
        for grp in groups:
            agg = _agg_time(error_df, col, grp)
            if not agg.empty:
                all_means.extend(agg["mean"].tolist())
                all_ci.extend(agg["ci_h"].tolist())
    agg_all = _agg_time(error_df)
    all_means.extend(agg_all["mean"].tolist())
    all_ci.extend(agg_all["ci_h"].tolist())

    y_lo = max(0, min(m - h for m, h in zip(all_means, all_ci)) * 0.92)
    y_hi = max(m + h for m, h in zip(all_means, all_ci)) * 1.08

    fig, axes = plt.subplots(2, 3,
                             figsize=(DOUBLE_COL * 1.4, FIG_HEIGHT * 2.3),
                             sharey=True,
                             constrained_layout=True)
    axes_flat = axes.flatten()

    for idx, (col, lbl, order) in enumerate(panels):
        ax     = axes_flat[idx]
        groups = order or sorted(error_df[col].dropna().unique())
        row, col_pos = divmod(idx, 3)

        for i, grp in enumerate(groups):
            agg = _agg_time(error_df, col, grp)
            if agg.empty:
                continue
            c = WONG[i % len(WONG)]
            ax.plot(agg["Month"], agg["mean"], color=c,
                    linewidth=1.4, marker="o", markersize=2.5,
                    label=str(grp), zorder=3)
            ax.fill_between(agg["Month"],
                            agg["mean"] - agg["ci_h"],
                            agg["mean"] + agg["ci_h"],
                            alpha=0.14, color=c, zorder=2)

        _set_month_ticks(ax, months)
        ax.set_ylim(y_lo, y_hi)
        if row == 1:
            ax.set_xlabel("Follow-up (months)", fontsize=8)
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
        if col_pos == 0:
            ax.set_ylabel(METRIC_LBL, fontsize=8)
        ax.legend(title=lbl, title_fontsize=7, fontsize=6.5,
                  framealpha=0.85, handlelength=1.2,
                  borderpad=0.3, labelspacing=0.2,
                  loc="best")
        _panel_label(ax, letters[idx])

    # Panel (f): overall
    ax  = axes_flat[5]
    agg = _agg_time(error_df)
    ax.plot(agg["Month"], agg["mean"], color=WONG[0],
            linewidth=1.8, marker="o", markersize=3,
            label="All subjects", zorder=3)
    ax.fill_between(agg["Month"],
                    agg["mean"] - agg["ci_h"],
                    agg["mean"] + agg["ci_h"],
                    alpha=0.15, color=WONG[0], zorder=2)
    _set_month_ticks(ax, months)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Follow-up (months)", fontsize=8)
    ax.legend(title="Overall", title_fontsize=7, fontsize=6.5,
              framealpha=0.85, handlelength=1.2)
    _panel_label(ax, "f")

    _save(fig, "fig1_mae_vs_time_combined")


# ── Figure 2: forest plot — mean MAE ± 95 % CI ───────────────────────────────

def plot_forest(mae_tbl: pd.DataFrame) -> None:
    """
    Horizontal dot-and-whisker (forest plot) of mean MAE [95 % CI].
    Stratum labels rendered as in-axis annotations on the left; group
    labels on the y-axis.  Overall mean shown as a dashed reference.
    """
    overall_mean = mae_tbl.loc[mae_tbl["Stratum"] == "Overall", "Mean_MAE"].iloc[0]

    df = mae_tbl[
        (mae_tbl["Stratum"] != "Overall") &
        ~((mae_tbl["Stratum"] == "APOE4") & (mae_tbl["Group"] == "Unknown"))
    ].copy()

    stratum_order = ["Diagnosis", "Sex", "APOE4", "Age_group", "Study", "Fold"]
    stratum_label = {"Diagnosis": "Diagnosis", "Sex": "Sex", "APOE4": "APOE4",
                     "Age_group": "Age group",  "Study": "Study", "Fold": "Fold"}
    df["Stratum"] = pd.Categorical(df["Stratum"], categories=stratum_order, ordered=True)
    df = df.sort_values(["Stratum", "Group"]).reset_index(drop=True)
    df["y"] = range(len(df))

    stratum_colors = {s: WONG[i] for i, s in enumerate(stratum_order)}
    df["color"]    = df["Stratum"].map(stratum_colors)

    fig_h = max(4.5, len(df) * 0.30)
    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.92, fig_h),
                           constrained_layout=True)

    # Shaded bands + stratum labels
    stratum_starts = df.groupby("Stratum", observed=True)["y"].min()
    stratum_ends   = df.groupby("Stratum", observed=True)["y"].max()
    x_label_pos    = ax.get_xlim()[0] if ax.get_xlim()[0] != 0.0 else \
                     df["CI_lower"].min() - 0.01
    for s in stratum_order:
        if s not in stratum_starts.index:
            continue
        y0, y1 = stratum_starts[s] - 0.5, stratum_ends[s] + 0.5
        ax.axhspan(y0, y1, alpha=0.07, color=stratum_colors[s], zorder=0)
        # Stratum label centred in band, left of plot
        ax.text(0.01, (y0 + y1) / 2, stratum_label[s],
                transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=7,
                color=stratum_colors[s], fontweight="bold")

    # Error bars
    for _, r in df.iterrows():
        ax.errorbar(r["Mean_MAE"], r["y"],
                    xerr=[[r["Mean_MAE"] - r["CI_lower"]],
                          [r["CI_upper"] - r["Mean_MAE"]]],
                    fmt="o", color=r["color"],
                    capsize=2.5, capthick=0.8,
                    markersize=4, linewidth=0.9, zorder=3)

    # Overall reference line
    ax.axvline(overall_mean, color="#444444", linestyle="--",
               linewidth=1.0, zorder=1,
               label=f"Overall mean  {overall_mean:.3f}")

    # y-axis: group label + n
    ylabels = [f"  {r['Group']}  (n={int(r['N']):,})" for _, r in df.iterrows()]
    ax.set_yticks(df["y"])
    ax.set_yticklabels(ylabels, fontsize=7.5)
    ax.invert_yaxis()

    ax.set_xlabel("Mean MAE  [95 % CI]", fontsize=9)
    ax.legend(framealpha=0.9, handlelength=1.5,
              loc="lower right", fontsize=7.5)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    _save(fig, "fig2_forest_mae_ci")


# ── Figure 3: per-covariate single-column line plots ─────────────────────────

def plot_line_per_covariate(error_df: pd.DataFrame) -> None:
    """One publication-ready single-column figure per covariate."""
    specs = [
        ("Diagnosis", "Diagnosis",  ["CN", "MCI", "AD", "Other"]),
        ("Sex",       "Sex",        ["Female", "Male"]),
        ("APOE4",     "APOE4",      ["ε3/ε3", "1 ε4 allele", "2 ε4 alleles"]),
        ("Age_group", "Age group",  ["<60", "60–70", "70–80", "≥80"]),
        ("Study",     "Study",      None),
        ("Fold",      "Fold",       [0, 1, 2, 3, 4]),
    ]
    months = _observed_months(error_df)

    for col, lbl, order in specs:
        groups = order or sorted(error_df[col].dropna().unique())
        fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.2, FIG_HEIGHT),
                               constrained_layout=True)

        for i, grp in enumerate(groups):
            agg = _agg_time(error_df, col, grp)
            if agg.empty:
                continue
            c = WONG[i % len(WONG)]
            ax.plot(agg["Month"], agg["mean"], color=c,
                    linewidth=1.5, marker="o", markersize=3,
                    label=str(grp), zorder=3)
            ax.fill_between(agg["Month"],
                            agg["mean"] - agg["ci_h"],
                            agg["mean"] + agg["ci_h"],
                            alpha=0.14, color=c, zorder=2)

        _set_month_ticks(ax, months)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Follow-up (months)", fontsize=8)
        ax.set_ylabel(METRIC_LBL, fontsize=8)
        ax.legend(title=lbl, title_fontsize=7.5, fontsize=7,
                  framealpha=0.85, handlelength=1.3,
                  borderpad=0.35, labelspacing=0.2, loc="best")

        _save(fig, f"fig3_mae_vs_time_{col.lower()}")


# ── Figure 4: signed error (bias) vs Time — 1×2 panel ───────────────────────

def plot_bias_vs_time(error_df: pd.DataFrame) -> None:
    """
    1×2 figure: (a) Bias vs month by Diagnosis, (b) Bias vs month by Sex.
    Dashed zero line marks perfect calibration.
    """
    panels = [
        ("Diagnosis", ["CN", "MCI", "AD", "Other"]),
        ("Sex",       ["Female", "Male"]),
    ]
    months = _observed_months(error_df)

    fig, axes = plt.subplots(1, 2,
                             figsize=(DOUBLE_COL, FIG_HEIGHT),
                             sharey=True,
                             constrained_layout=True)

    for ax, (col, order) in zip(axes, panels):
        for i, grp in enumerate(order):
            sub = error_df[error_df[col] == grp].dropna(subset=["Bias"])

            def _ci(x):
                m, lo, hi = _mean_ci(x.values)
                return pd.Series({"mean": m, "ci_h": (hi - lo) / 2})

            agg = (sub.groupby("Month")["Bias"]
                      .apply(_ci)
                      .reset_index()
                      .rename(columns={"level_1": "stat"})
                      .pivot(index="Month", columns="stat", values="Bias")
                      .reset_index())
            if agg.empty:
                continue
            c = WONG[i % len(WONG)]
            ax.plot(agg["Month"], agg["mean"], color=c,
                    linewidth=1.5, marker="o", markersize=2.5,
                    label=str(grp), zorder=3)
            ax.fill_between(agg["Month"],
                            agg["mean"] - agg["ci_h"],
                            agg["mean"] + agg["ci_h"],
                            alpha=0.14, color=c, zorder=2)

        ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--", zorder=1)
        _set_month_ticks(ax, months)
        ax.set_xlabel("Follow-up (months)", fontsize=8)
        ax.legend(title=col, title_fontsize=7, fontsize=6.5,
                  framealpha=0.85, handlelength=1.2,
                  borderpad=0.3, labelspacing=0.2)

    axes[0].set_ylabel("Bias  mean(predicted − real)  (a.u.)", fontsize=8)
    _panel_label(axes[0], "a")
    _panel_label(axes[1], "b")

    _save(fig, "fig4_bias_vs_time")


# ── Figure 5: violin of MAE distribution per Diagnosis ───────────────────────

def plot_violin_diagnosis(error_df: pd.DataFrame) -> None:
    """
    Violin + strip of MAE distributions per diagnostic group.
    White diamond = mean ± 95 % CI.
    """
    order   = ["CN", "MCI", "AD", "Other"]
    df      = error_df[error_df["Diagnosis"].isin(order)].dropna(subset=[METRIC])
    palette = {g: WONG[i] for i, g in enumerate(order)}

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.1, FIG_HEIGHT),
                           constrained_layout=True)

    sns.violinplot(data=df, x="Diagnosis", y=METRIC, order=order,
                   palette=palette, inner=None, linewidth=0.7,
                   saturation=0.80, ax=ax)
    sns.stripplot(data=df, x="Diagnosis", y=METRIC, order=order,
                  palette=palette, size=1.0, alpha=0.30,
                  jitter=True, linewidth=0, ax=ax)

    for i, grp in enumerate(order):
        vals = df[df["Diagnosis"] == grp][METRIC].dropna().values
        if len(vals) < 2:
            continue
        m, lo, hi = _mean_ci(vals)
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]],
                    fmt="D", color="white", ecolor="black",
                    markersize=4.5, linewidth=1.2, capsize=3, zorder=5)

    ax.set_xlabel("Diagnosis", fontsize=8)
    ax.set_ylabel(METRIC_LBL, fontsize=8)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="x", length=0)

    _save(fig, "fig5_violin_mae_diagnosis")


# ── Figure 6: heatmap — mean MAE (Diagnosis / Age group) × month bin ─────────

def plot_heatmap_overview(error_df: pd.DataFrame) -> None:
    """
    2-row heatmap: rows = covariate groups, columns = 12-month time bins.
    Shared colour scale so both panels are directly comparable.
    """
    specs = [
        ("Diagnosis", "Diagnosis", ["CN", "MCI", "AD", "Other"]),
        ("Age_group", "Age group", ["<60", "60–70", "70–80", "≥80"]),
    ]

    bin_width = 12
    max_month = int(error_df["Month"].max())
    bins      = np.arange(0, max_month + bin_width, bin_width)
    bin_lbls  = ((bins[:-1] + bins[1:]) / 2).astype(int)
    edf = error_df.copy()
    edf["Month_bin"] = pd.cut(edf["Month"], bins=bins,
                              labels=bin_lbls, include_lowest=True).astype(float)

    # Shared colour range
    global_vmin, global_vmax = np.inf, -np.inf
    pivots = []
    for col, _, order in specs:
        p = edf.pivot_table(values=METRIC, index=col,
                            columns="Month_bin", aggfunc="mean")
        if order:
            p = p.reindex([r for r in order if r in p.index])
        pivots.append(p)
        global_vmin = min(global_vmin, p.min().min())
        global_vmax = max(global_vmax, p.max().max())

    fig, axes = plt.subplots(len(specs), 1,
                             figsize=(DOUBLE_COL * 0.92, FIG_HEIGHT * len(specs) + 0.4),
                             constrained_layout=True)

    for ax, (col, lbl, order), pivot in zip(axes, specs, pivots):
        annot = pivot.applymap(lambda v: f"{v:.3f}" if pd.notna(v) else "")
        sns.heatmap(pivot, cmap="YlOrRd", annot=annot, fmt="",
                    linewidths=0.3, linecolor="#e0e0e0",
                    vmin=global_vmin, vmax=global_vmax,
                    cbar_kws={"label": METRIC_LBL, "shrink": 0.75},
                    annot_kws={"fontsize": 7},
                    ax=ax)
        ax.set_xlabel("Month bin centre" if ax == axes[-1] else "",
                      fontsize=8)
        ax.set_ylabel(lbl, fontsize=8)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", rotation=0,  labelsize=7.5)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(METRIC_LBL, fontsize=7.5)

    _save(fig, "fig6_heatmap_mae")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("STEP 1 — Parsing and merging fold trajectory files")
    print("=" * 70)
    merged = merge_fold_files(PRED_DIR, MERGED_CSV)

    print("\n" + "=" * 70)
    print("STEP 2 — Loading covariates")
    print("=" * 70)
    cov = pd.read_csv(COV_FILE)
    print(f"  {len(cov):,} rows, {cov['PTID'].nunique()} subjects")

    print("\n" + "=" * 70)
    print("STEP 3 — Building error table  (aggregate across ROIs)")
    print("=" * 70)
    error_df = build_error_table(merged, cov)

    print("\n" + "=" * 70)
    print("STEP 4 — MAE + 95 % CI report")
    print("=" * 70)
    mae_tbl = report_mae_ci(error_df)

    print("=" * 70)
    print("STEP 5 — Generating NeurIPS figures")
    print("=" * 70)
    plot_combined_overview(error_df)
    plot_forest(mae_tbl)
    plot_line_per_covariate(error_df)
    plot_bias_vs_time(error_df)
    plot_violin_diagnosis(error_df)
    plot_heatmap_overview(error_df)

    print("\n" + "=" * 70)
    print("DONE — all outputs in", OUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
