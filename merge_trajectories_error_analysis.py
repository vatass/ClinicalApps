"""
NeuRIPS 2026: Merge predicted trajectories and analyse prediction error
stratified by longitudinal covariates.

Steps
-----
1. Merge all per-subject trajectory CSVs in predictions/ into one combined CSV.
2. At each subject's observed time-point (covariate file), extract the
   predicted value and the change-from-baseline (delta = pred(T) - pred(0))
   for every ROI; summarise as mean |delta| (MAE proxy) per observation.
3. Report overall and stratified mean MAE with 95% CI (t-distribution).
4. Produce NeurIPS-quality figures: error vs Time per covariate.
"""

import os
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", message=".*seaborn styles.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ── Paths ─────────────────────────────────────────────────────────────────────
PRED_DIR   = "./predictions"
COV_FILE   = "./longitudinal_covariates_allstudies.csv"
OUT_DIR    = "./trajectory_error_analysis"
MERGED_CSV = os.path.join(OUT_DIR, "merged_trajectories.csv")
ERROR_CSV  = os.path.join(OUT_DIR, "error_by_covariates.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# ── NeurIPS plot style ────────────────────────────────────────────────────────
# NeurIPS: 10pt base font, single-column ≈ 3.25 in, double-column ≈ 6.75 in
SINGLE_COL = 3.25
DOUBLE_COL = 6.75
FIG_HEIGHT = 2.6          # compact height for paper figures

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    7.5,
    "legend.title_fontsize": 7.5,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.major.size":   3,
    "ytick.major.size":   3,
    "lines.linewidth":    1.5,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linewidth":     0.5,
})

# Wong (2011) colorblind-safe palette — 7 distinguishable colours
WONG = [
    "#0072B2",   # blue
    "#D55E00",   # vermillion
    "#009E73",   # bluish green
    "#CC79A7",   # reddish purple
    "#E69F00",   # orange
    "#56B4E9",   # sky blue
    "#000000",   # black
]

METRIC     = "mean_abs_delta"          # column name (= |z_pred − z_DLICV|)
METRIC_LBL = "Prediction Error  |z-pred − z-DLICV|  (SD)"


# ─────────────────────────────────────────────────────────────────────────────
# 1. MERGE ALL TRAJECTORY CSV FILES
# ─────────────────────────────────────────────────────────────────────────────

def merge_trajectories(pred_dir: str, out_path: str) -> pd.DataFrame:
    """
    Read every predictions/trajectory_ptid_*.csv, prepend a PTID column,
    and concatenate into a single wide DataFrame.

    Output columns: PTID, ROI_Index, Month_0 … Month_119
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
        df   = pd.read_csv(os.path.join(pred_dir, fname))
        frames.append(pd.concat([pd.DataFrame({"PTID": [ptid] * len(df)}), df], axis=1))

    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(out_path, index=False)
    print(f"[merge] Saved {len(merged):,} rows → {out_path}")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 2. BUILD ERROR TABLE
# ─────────────────────────────────────────────────────────────────────────────

def _clean_diagnosis(dx: str) -> str:
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
    True prediction error at each observed time-point T:

        error(T) = | z_pred(T)  −  z_DLICV(T) |

    where both quantities are z-scored using the *baseline* (T=0) population
    statistics so they are on the same scale:

        z_pred(T)  = ( pred_mean(T)  − μ_pred0  ) / σ_pred0
        z_DLICV(T) = ( DLICV(T)      − μ_DLICV0 ) / σ_DLICV0

    pred_mean(T) = mean over 145 ROIs of the predicted trajectory at month T.
    DLICV        = differential local intracranial volume (actual observed).
    Baseline correlation pred_mean(0) ↔ DLICV(0):  r ≈ 0.877.
    """
    month_cols = [c for c in merged.columns if re.match(r"^Month_\d+$", c)]
    max_month  = max(int(c.split("_")[1]) for c in month_cols)

    common = set(merged["PTID"].unique()) & set(cov["PTID"].unique())
    print(f"[error] Subjects in both files: {len(common)}")

    # ── Baseline population statistics for z-scoring ──────────────────────────
    # pred_mean at T=0: mean over all 145 ROIs per subject
    pred_mean_0 = (
        merged.groupby("PTID")["Month_0"].mean()
        .reset_index().rename(columns={"Month_0": "pred_mean_0"})
    )
    mu_pred0  = pred_mean_0["pred_mean_0"].mean()
    std_pred0 = pred_mean_0["pred_mean_0"].std()
    print(f"[error] pred_mean baseline: μ={mu_pred0:.4f}, σ={std_pred0:.4f}")

    # DLICV at T=0 per subject (earliest visit)
    dlicv_0 = (
        cov[cov["PTID"].isin(common)].sort_values("Time")
        .groupby("PTID")["DLICV"].first()
        .reset_index().rename(columns={"DLICV": "dlicv_0"})
    )
    mu_dlicv0  = dlicv_0["dlicv_0"].mean()
    std_dlicv0 = dlicv_0["dlicv_0"].std()
    print(f"[error] DLICV   baseline: μ={mu_dlicv0:.1f}, σ={std_dlicv0:.1f}")

    # ── Build per-observation error ───────────────────────────────────────────
    cov_sub = cov[cov["PTID"].isin(common)].copy()
    cov_sub["Diagnosis"] = cov_sub["Diagnosis"].apply(_clean_diagnosis)
    cov_sub["Age_group"] = cov_sub["Age"].apply(_age_group)
    cov_sub["Sex"]       = cov_sub["Sex"].map({0: "Female", 1: "Male"}).fillna("Unknown")
    cov_sub["APOE4"]     = cov_sub["APOE4_Alleles"].map(
        {0.0: "ε3/ε3", 1.0: "1 ε4 allele", 2.0: "2 ε4 alleles", -1.0: "Unknown"}
    ).fillna("Unknown")

    # Pre-compute pred_mean per (PTID, month) — only the columns we need
    traj_by_ptid = {}
    for ptid, grp in merged.groupby("PTID"):
        # store as dict {month_col: mean_value}
        traj_by_ptid[ptid] = {
            col: grp[col].mean() for col in month_cols
        }

    records = []
    for _, row in cov_sub.iterrows():
        ptid = row["PTID"]
        t    = int(row["Time"])
        dlicv_obs = row["DLICV"]

        if (t > max_month or ptid not in traj_by_ptid
                or pd.isna(dlicv_obs)):
            continue
        month_col = f"Month_{t}"
        if month_col not in traj_by_ptid[ptid]:
            continue

        pred_mean_T = traj_by_ptid[ptid][month_col]

        # Z-score using baseline population statistics
        z_pred  = (pred_mean_T - mu_pred0)  / std_pred0
        z_dlicv = (dlicv_obs   - mu_dlicv0) / std_dlicv0

        # True prediction error (in SD units, scale-free)
        pred_error  = z_pred - z_dlicv          # signed
        abs_error   = abs(pred_error)

        records.append({
            "PTID":           ptid,
            "Time":           t,
            "z_pred":         z_pred,
            "z_dlicv":        z_dlicv,
            "pred_error":     pred_error,        # signed: + means over-prediction
            "mean_abs_delta": abs_error,         # keep same column name for plots
            "Diagnosis":      row["Diagnosis"],
            "Sex":            row["Sex"],
            "APOE4":          row["APOE4"],
            "Age_group":      row["Age_group"],
            "Study":          row["Study"],
            "Age":            row["Age"],
        })

    error_df = pd.DataFrame(records)
    error_df.to_csv(ERROR_CSV, index=False)
    print(f"[error] Saved {len(error_df):,} rows → {ERROR_CSV}")
    return error_df


# ─────────────────────────────────────────────────────────────────────────────
# 3. MAE + 95 % CI REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def _mean_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    """Return (mean, ci_lower, ci_upper) using the t-distribution."""
    n  = len(values)
    m  = values.mean()
    if n < 2:
        return m, m, m
    se    = stats.sem(values, nan_policy="omit")
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    h = t_crit * se
    return m, m - h, m + h


def report_mae_ci(error_df: pd.DataFrame) -> pd.DataFrame:
    """
    Print and return a tidy table of mean MAE ± 95 % CI, overall and
    stratified by Diagnosis, Sex, APOE4, Age group, and Study.
    """
    metric = METRIC
    rows   = []

    # Overall
    vals = error_df[metric].dropna().values
    m, lo, hi = _mean_ci(vals)
    rows.append({
        "Stratum": "Overall", "Group": "—",
        "N": len(vals), "Mean MAE": m,
        "CI_lower": lo, "CI_upper": hi,
    })

    covariates = [
        ("Diagnosis", ["CN", "MCI", "AD", "Other"]),
        ("Sex",       ["Female", "Male"]),
        ("APOE4",     ["ε3/ε3", "1 ε4 allele", "2 ε4 alleles", "Unknown"]),
        ("Age_group", ["<60", "60–70", "70–80", "≥80"]),
        ("Study",     None),
    ]

    for col, order in covariates:
        groups = order or sorted(error_df[col].dropna().unique())
        for grp in groups:
            vals = error_df[error_df[col] == grp][metric].dropna().values
            if len(vals) == 0:
                continue
            m, lo, hi = _mean_ci(vals)
            rows.append({
                "Stratum": col, "Group": grp,
                "N": len(vals), "Mean MAE": m,
                "CI_lower": lo, "CI_upper": hi,
            })

    tbl = pd.DataFrame(rows)

    width = 72
    print("\n" + "=" * width)
    print("PREDICTION ERROR MAE — MEAN ± 95 % CI (t-distribution)")
    print(f"  Metric : |z_pred(T) − z_DLICV(T)|  (SD units)")
    print(f"           pred_mean(T) = mean over 145 ROIs of trajectory at month T")
    print(f"           both z-scored using baseline (T=0) population μ and σ")
    print(f"  CI     : 95 % confidence interval on the mean (t-distribution)")
    print("=" * width)
    print(f"  {'Stratum':<12} {'Group':<22} {'N':>6}  {'Mean MAE':>10}  "
          f"{'95 % CI':>22}")
    print("  " + "-" * (width - 2))

    prev_stratum = None
    for _, r in tbl.iterrows():
        if r["Stratum"] != prev_stratum:
            if prev_stratum is not None:
                print()
            prev_stratum = r["Stratum"]
        print(f"  {r['Stratum']:<12} {str(r['Group']):<22} {int(r['N']):>6}  "
              f"{r['Mean MAE']:>10.4f}  "
              f"[{r['CI_lower']:.4f}, {r['CI_upper']:.4f}]")

    print("=" * width)
    tbl.to_csv(os.path.join(OUT_DIR, "mae_ci_summary.csv"), index=False)
    print(f"[report] Saved: {OUT_DIR}/mae_ci_summary.csv\n")
    return tbl


# ─────────────────────────────────────────────────────────────────────────────
# 4. VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def _bin_time(df: pd.DataFrame, bin_width: int = 12) -> pd.DataFrame:
    df   = df.copy()
    bins = np.arange(0, df["Time"].max() + bin_width, bin_width)
    lbls = (bins[:-1] + bins[1:]) / 2
    df["Time_bin"] = pd.cut(df["Time"], bins=bins, labels=lbls,
                            include_lowest=True).astype(float)
    return df


def _agg_ci(df: pd.DataFrame, time_col: str = "Time_bin") -> pd.DataFrame:
    """Aggregate metric by time bin → mean + 95 % CI half-width."""
    def _ci_row(x):
        m, lo, hi = _mean_ci(x.dropna().values)
        return pd.Series({"mean": m, "ci_h": (hi - lo) / 2})

    return (df.groupby(time_col)[METRIC]
              .apply(_ci_row)
              .reset_index()
              .rename(columns={"level_1": "stat"})
              .pivot(index=time_col, columns="stat", values=METRIC)
              .reset_index()
              .rename(columns={time_col: "Time_bin"}))


def _save(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "png"):
        fpath = os.path.join(OUT_DIR, f"{stem}.{ext}")
        fig.savefig(fpath)
        print(f"[plot] Saved: {fpath}")
    plt.close(fig)


# ── Figure 1: Combined 2 × 3 overview ────────────────────────────────────────

def plot_combined_overview(error_df: pd.DataFrame) -> None:
    """2 × 3 panel: error vs time for five covariates + overall."""
    panels = [
        ("Diagnosis", "Diagnosis",    ["CN", "MCI", "AD", "Other"]),
        ("Sex",       "Sex",          ["Female", "Male"]),
        ("APOE4",     "APOE4",        ["ε3/ε3", "1 ε4 allele", "2 ε4 alleles", "Unknown"]),
        ("Age_group", "Age group",    ["<60", "60–70", "70–80", "≥80"]),
        ("Study",     "Study",        None),
    ]

    df   = _bin_time(error_df, bin_width=12)
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COL * 1.35, FIG_HEIGHT * 2.2),
                             constrained_layout=True)
    axes_flat = axes.flatten()

    for idx, (col, lbl, order) in enumerate(panels):
        ax     = axes_flat[idx]
        groups = order or sorted(df[col].dropna().unique())

        for i, grp in enumerate(groups):
            sub = df[df[col] == grp]
            agg = _agg_ci(sub)
            if agg.empty:
                continue
            c = WONG[i % len(WONG)]
            ax.plot(agg["Time_bin"], agg["mean"], color=c,
                    linewidth=1.5, marker="o", markersize=2.5,
                    label=str(grp), zorder=3)
            ax.fill_between(agg["Time_bin"],
                            agg["mean"] - agg["ci_h"],
                            agg["mean"] + agg["ci_h"],
                            alpha=0.12, color=c, zorder=2)

        ax.set_xlabel("Time (months)")
        ax.set_ylabel(METRIC_LBL)
        ax.set_title(lbl, fontweight="bold")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(24))
        ax.legend(framealpha=0.85, handlelength=1.4, borderpad=0.4,
                  labelspacing=0.25)

    # Panel 6: overall
    ax   = axes_flat[5]
    agg  = _agg_ci(df)
    ax.plot(agg["Time_bin"], agg["mean"], color=WONG[0],
            linewidth=1.8, marker="o", markersize=3,
            label="All subjects", zorder=3)
    ax.fill_between(agg["Time_bin"],
                    agg["mean"] - agg["ci_h"],
                    agg["mean"] + agg["ci_h"],
                    alpha=0.15, color=WONG[0], zorder=2)
    ax.set_xlabel("Time (months)")
    ax.set_ylabel(METRIC_LBL)
    ax.set_title("Overall", fontweight="bold")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(24))
    ax.legend(framealpha=0.85, handlelength=1.4)

    _save(fig, "fig1_error_vs_time_combined")


# ── Figure 2: Forest plot — mean MAE ± 95 % CI per covariate group ───────────

def plot_forest(mae_tbl: pd.DataFrame) -> None:
    """
    Horizontal dot-and-whisker (forest) plot of mean MAE [95 % CI] for every
    covariate group, grouped by stratum with a vertical reference line at the
    overall mean.
    """
    overall_mean = mae_tbl.loc[mae_tbl["Stratum"] == "Overall", "Mean MAE"].iloc[0]

    # Drop "Overall" row and "Unknown" APOE4 for cleaner display
    df = mae_tbl[
        (mae_tbl["Stratum"] != "Overall") &
        ~((mae_tbl["Stratum"] == "APOE4") & (mae_tbl["Group"] == "Unknown"))
    ].copy()

    # Build display label and group colour
    stratum_order = ["Diagnosis", "Sex", "APOE4", "Age_group", "Study"]
    stratum_label = {
        "Diagnosis": "Diagnosis", "Sex": "Sex", "APOE4": "APOE4",
        "Age_group": "Age group", "Study": "Study",
    }
    df["Stratum"] = pd.Categorical(df["Stratum"], categories=stratum_order, ordered=True)
    df = df.sort_values(["Stratum", "Group"]).reset_index(drop=True)
    df["y"] = range(len(df))

    stratum_colors = {s: WONG[i] for i, s in enumerate(stratum_order)}
    df["color"]    = df["Stratum"].map(stratum_colors)

    fig_h = max(3.0, len(df) * 0.28)
    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.88, fig_h),
                           constrained_layout=True)

    for _, r in df.iterrows():
        ax.errorbar(r["Mean MAE"], r["y"],
                    xerr=[[r["Mean MAE"] - r["CI_lower"]],
                          [r["CI_upper"] - r["Mean MAE"]]],
                    fmt="o", color=r["color"],
                    capsize=2.5, capthick=0.8,
                    markersize=4, linewidth=0.9, zorder=3)

    # Reference line: overall mean
    ax.axvline(overall_mean, color="#555555", linestyle="--",
               linewidth=0.9, zorder=1, label=f"Overall mean ({overall_mean:.3f})")

    # y-axis labels: "Group (N=...)"
    ylabels = [f"{r['Group']}  (n={int(r['N']):,})" for _, r in df.iterrows()]
    ax.set_yticks(df["y"])
    ax.set_yticklabels(ylabels, fontsize=7.5)
    ax.invert_yaxis()

    # Stratum separator lines + labels
    stratum_starts = df.groupby("Stratum", observed=True)["y"].min()
    stratum_ends   = df.groupby("Stratum", observed=True)["y"].max()
    for s in stratum_order:
        if s not in stratum_starts.index:
            continue
        y0, y1 = stratum_starts[s] - 0.5, stratum_ends[s] + 0.5
        ax.axhspan(y0, y1, alpha=0.06, color=stratum_colors[s], zorder=0)
        ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] != 0 else
                df["CI_lower"].min() * 0.97,
                (y0 + y1) / 2, stratum_label[s],
                va="center", ha="right", fontsize=7,
                color=stratum_colors[s], fontweight="bold",
                transform=ax.get_yaxis_transform())

    ax.set_xlabel(f"Mean MAE  (95 % CI)")
    ax.legend(framealpha=0.85, handlelength=1.5, loc="lower right",
              fontsize=7.5)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax.grid(axis="y", visible=False)

    _save(fig, "fig2_forest_mae_ci")


# ── Figure 3: per-covariate line plots (one per covariate, single-column) ────

def plot_line_per_covariate(error_df: pd.DataFrame) -> None:
    """Individual single-column line plots for each covariate."""
    specs = [
        ("Diagnosis", "Diagnosis",  ["CN", "MCI", "AD", "Other"]),
        ("Sex",       "Sex",        ["Female", "Male"]),
        ("APOE4",     "APOE4",      ["ε3/ε3", "1 ε4 allele", "2 ε4 alleles"]),
        ("Age_group", "Age group",  ["<60", "60–70", "70–80", "≥80"]),
        ("Study",     "Study",      None),
    ]

    df = _bin_time(error_df, bin_width=12)

    for col, lbl, order in specs:
        groups = order or sorted(df[col].dropna().unique())
        fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.18, FIG_HEIGHT),
                               constrained_layout=True)

        for i, grp in enumerate(groups):
            sub = df[df[col] == grp]
            agg = _agg_ci(sub)
            if agg.empty:
                continue
            c = WONG[i % len(WONG)]
            ax.plot(agg["Time_bin"], agg["mean"], color=c,
                    linewidth=1.5, marker="o", markersize=2.5,
                    label=str(grp), zorder=3)
            ax.fill_between(agg["Time_bin"],
                            agg["mean"] - agg["ci_h"],
                            agg["mean"] + agg["ci_h"],
                            alpha=0.13, color=c, zorder=2)

        ax.set_xlabel("Time (months)")
        ax.set_ylabel(METRIC_LBL)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(24))
        ax.legend(title=lbl, framealpha=0.85, handlelength=1.4,
                  borderpad=0.4, labelspacing=0.25)

        _save(fig, f"fig3_error_vs_time_{col.lower()}")


# ── Figure 4: heatmap — mean MAE in (covariate × time-bin) ───────────────────

def plot_heatmap_overview(error_df: pd.DataFrame) -> None:
    """
    2-row × 1-column figure: Diagnosis × Time and Age group × Time heatmaps
    stacked vertically (double-column width).
    """
    specs = [
        ("Diagnosis", "Diagnosis", ["CN", "MCI", "AD", "Other"]),
        ("Age_group", "Age group", ["<60", "60–70", "70–80", "≥80"]),
    ]

    df   = _bin_time(error_df, bin_width=24)
    fig, axes = plt.subplots(len(specs), 1,
                             figsize=(DOUBLE_COL * 0.88, FIG_HEIGHT * len(specs)),
                             constrained_layout=True)

    for ax, (col, lbl, order) in zip(axes, specs):
        pivot = df.pivot_table(
            values=METRIC, index=col, columns="Time_bin", aggfunc="mean"
        )
        if order:
            pivot = pivot.reindex([r for r in order if r in pivot.index])

        # format annotation values
        annot = pivot.map(lambda v: f"{v:.3f}" if not np.isnan(v) else "")

        sns.heatmap(
            pivot, cmap="YlOrRd", annot=annot, fmt="",
            linewidths=0.3, linecolor="#dddddd",
            cbar_kws={"label": METRIC_LBL, "shrink": 0.8},
            ax=ax, vmin=0,
        )
        ax.set_xlabel("Time bin centre (months)")
        ax.set_ylabel(lbl)
        ax.set_title(lbl, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
        # fix colorbar font
        ax.collections[0].colorbar.ax.tick_params(labelsize=7)
        ax.collections[0].colorbar.set_label(METRIC_LBL, fontsize=8)

    _save(fig, "fig4_heatmap_diagnosis_age")


# ── Figure 5: violin + strip — MAE distribution per diagnosis ────────────────

def plot_violin_diagnosis(error_df: pd.DataFrame) -> None:
    """
    Violin plot + strip of MAE distributions per diagnostic group.
    Single-column width, NeurIPS-ready.
    """
    order = ["CN", "MCI", "AD", "Other"]
    df    = error_df[error_df["Diagnosis"].isin(order)].copy()

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.05, FIG_HEIGHT),
                           constrained_layout=True)

    palette = {g: WONG[i] for i, g in enumerate(order)}
    sns.violinplot(data=df, x="Diagnosis", y=METRIC, order=order,
                   palette=palette, inner=None, linewidth=0.8,
                   saturation=0.85, ax=ax)
    sns.stripplot(data=df, x="Diagnosis", y=METRIC, order=order,
                  palette=palette, size=1.2, alpha=0.35, jitter=True,
                  linewidth=0, ax=ax)

    # overlay mean ± 95 % CI as error bar
    for i, grp in enumerate(order):
        vals = df[df["Diagnosis"] == grp][METRIC].dropna().values
        if len(vals) < 2:
            continue
        m, lo, hi = _mean_ci(vals)
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]],
                    fmt="D", color="white", ecolor="black",
                    markersize=4, linewidth=1.2, capsize=3, zorder=5)

    ax.set_xlabel("Diagnosis")
    ax.set_ylabel(METRIC_LBL)
    ax.xaxis.set_tick_params(length=0)

    _save(fig, "fig5_violin_mae_diagnosis")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def plot_signed_error_vs_time(error_df: pd.DataFrame) -> None:
    """
    Signed prediction error vs Time (bias plot).
    Positive = model over-predicts brain volume (z_pred > z_DLICV).
    Negative = model under-predicts.
    Stratified by Diagnosis.
    """
    order = ["CN", "MCI", "AD", "Other"]
    df    = _bin_time(error_df.dropna(subset=["pred_error"]), bin_width=12)

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.18, FIG_HEIGHT),
                           constrained_layout=True)

    for i, grp in enumerate(order):
        sub = df[df["Diagnosis"] == grp]

        def _ci_signed(x):
            m, lo, hi = _mean_ci(x.dropna().values)
            return pd.Series({"mean": m, "ci_h": (hi - lo) / 2})

        agg = (sub.groupby("Time_bin")["pred_error"]
               .apply(_ci_signed)
               .reset_index()
               .pivot(index="Time_bin", columns="level_1", values="pred_error")
               .reset_index()
               .rename(columns={"Time_bin": "Time_bin"}))
        if agg.empty:
            continue
        c = WONG[i % len(WONG)]
        ax.plot(agg["Time_bin"], agg["mean"], color=c, linewidth=1.5,
                marker="o", markersize=2.5, label=grp, zorder=3)
        ax.fill_between(agg["Time_bin"],
                        agg["mean"] - agg["ci_h"],
                        agg["mean"] + agg["ci_h"],
                        alpha=0.13, color=c, zorder=2)

    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Signed Error  z-pred − z-DLICV  (SD)")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(24))
    ax.legend(title="Diagnosis", framealpha=0.85, handlelength=1.4,
              borderpad=0.4, labelspacing=0.25)

    _save(fig, "fig6_signed_error_vs_time_diagnosis")


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
    plot_heatmap_overview(error_df)
    plot_violin_diagnosis(error_df)
    plot_signed_error_vs_time(error_df)

    print("\n" + "=" * 70)
    print("DONE — all outputs in", OUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
