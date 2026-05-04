'''
NeuRIPS 2026: MCI Progression Prediction utilizing the 145 RoC. 
'''
'''
Manuscript 1: Rate of Change in the 145 Brain Volumes, Classification of Progressors vs Non-Progressors
using the 145 brain volumes and the predicted rate of change of the 145 brain volumes. 
Results for the Section:Predicted Rate of Change as a Tool for Discriminating Progressors from Non-Progressors
'''
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, roc_curve, roc_auc_score, classification_report,
    precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.inspection import permutation_importance
from sklearn.exceptions import ConvergenceWarning
from scipy import stats
from textwrap import wrap
from sklearn.metrics import (
    accuracy_score, roc_curve, roc_auc_score, average_precision_score,
    classification_report, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.calibration import calibration_curve
from statsmodels.stats.multitest import multipletests
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=FutureWarning, module='seaborn')
warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn.svm._base')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*overflow encountered.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered.*')
warnings.filterwarnings('ignore', message='.*findfont: Font family.*not found.*')
warnings.filterwarnings('ignore', message='.*Font family.*not found.*')
warnings.filterwarnings('ignore', message='.*seaborn styles shipped by Matplotlib.*')

# FIX 1: use the non-deprecated style name
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'DejaVu Sans',
    'font.sans-serif': ['DejaVu Sans'],
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 16,
    'figure.dpi': 300
})

# Asymmetric classifier selection for stress-test comparison:
# RNN-AD uses its WORST classifier, DKGP uses its BEST.
# If DKGP's best still can't beat RNN-AD's worst significantly,
# it is a strong honest statement about the relative model quality.
RNN_CLASSIFIER  = 'Gradient Boosting'    # worst for RNN-AD  (AUC = 0.767 ± 0.040)
DKGP_CLASSIFIER = 'Logistic Regression'  # best  for DKGP    (AUC = 0.791 ± 0.022)
FIXED_CLASSIFIER = 'Random Forest'       # used for real upper bound ONLY


RNN_CLASSIFIER       = 'Logistic Regression'
DKGP_CLASSIFIER      = 'Logistic Regression'
FIXED_CLASSIFIER     = 'Logistic Regression'
BRAINGEN_CLASSIFIER  = 'Logistic Regression'



def plot_comprehensive_roc_curves(baseline_results, rnn_pred_results, dkgp_pred_results,
                                   real_results, save_prefix='comprehensive_comparison',
                                   braingen_pred_results=None):
    """
    Create publication-quality ROC figure for Results section.
    Shows: Baseline, RNN-AD, DKGP, BrainGenFlow (when provided), Real UB.
    Clean design: no grid, despined axes, AUC ± std in legend.
    """
    sns.set(style="white", context="talk")
    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    # Ordered from bottom to top visually; colors chosen for accessibility
    methods = [
        ('Baseline volumes',                baseline_results,      RNN_CLASSIFIER,       '#9B59B6', (5, 2),       1.8),
        ('RNN-AD predicted',                rnn_pred_results,      RNN_CLASSIFIER,       '#E74C3C', (4, 2, 1, 2), 2.0),
        ('DKGP predicted',                  dkgp_pred_results,     DKGP_CLASSIFIER,      '#1A5276', (3, 1, 1, 1), 2.0),
        ('BrainGenFlow predicted (ours)',   braingen_pred_results, BRAINGEN_CLASSIFIER,  '#1A9C6E', 'solid',       2.5),
        ('Real trajectories (upper bound)', real_results,          FIXED_CLASSIFIER,     '#7F8C8D', (2, 2),        1.8),
    ]

    for label, results, clf, color, dashes, lw in methods:
        if results is None:
            continue
        cr  = results[clf]
        fpr = cr['fpr']
        tpr = cr['tpr']
        auc = cr['roc_auc']
        std = cr['roc_auc_std']

        line, = ax.plot(fpr, tpr, color=color, linewidth=lw,
                        label=f'{label}  (AUC = {auc:.3f} ± {std:.3f})')
        if dashes != 'solid':
            line.set_dashes(dashes)

    # Chance line
    ax.plot([0, 1], [0, 1], color='#AAAAAA', linewidth=1.2,
            linestyle='--', alpha=0.7, label='Chance level')

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel('False Positive Rate', fontsize=15)
    ax.set_ylabel('True Positive Rate',  fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=13,
                   width=1.2, length=5, direction='out')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('normal')

    # Clean legend — lower right
    legend = ax.legend(
        title='n = 882 MCI subjects',
        title_fontsize=10,
        fontsize=10,
        loc='lower right',
        frameon=True,
        framealpha=0.92,
        edgecolor='#CCCCCC',
        handlelength=2.4,
        handletextpad=0.6
    )
    legend.get_title().set_color('#555555')

    ax.annotate(
        f'RNN-AD: {RNN_CLASSIFIER} | DKGP: {DKGP_CLASSIFIER} | '
        f'BrainGenFlow: {BRAINGEN_CLASSIFIER} | Real UB: {FIXED_CLASSIFIER}',
        xy=(0.99, 0.01), xycoords='axes fraction',
        fontsize=7, color='#888888',
        ha='right', va='bottom', style='italic'
    )

    sns.despine()
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    plt.tight_layout()
    os.makedirs('./mciprogression', exist_ok=True)
    plt.savefig(f'./mciprogression/RNNAD_DKGP_{save_prefix}_comprehensive_roc_curves.png',
                dpi=600, bbox_inches='tight')
    plt.savefig(f'./mciprogression/RNNAD_DKGP_{save_prefix}_comprehensive_roc_curves.pdf',
                bbox_inches='tight')
    plt.savefig(f'./mciprogression/RNNAD_DKGP_{save_prefix}_comprehensive_roc_curves.svg',
                bbox_inches='tight')
    plt.close()
    braingen_note = f' | BrainGenFlow [{BRAINGEN_CLASSIFIER}]' if braingen_pred_results else ''
    print(f"✓ ROC curves saved — conditions: "
          f"Baseline [{RNN_CLASSIFIER}] | RNN-AD [{RNN_CLASSIFIER}] | "
          f"DKGP [{DKGP_CLASSIFIER}]{braingen_note} | Real UB [{FIXED_CLASSIFIER}]")



def load_baseline_volumes():
    """Load baseline raw volumes for 145 ROIs from the first timepoint of each subject."""
    print("Loading baseline volumes...")
    try:
        df = pd.read_csv('./manuscript1/HarmonizedROIVolumes.csv')
        volume_cols = [col for col in df.columns if col.startswith('y_H_MUSE_Volume')]
        baseline_data = []
        for subject in df['id'].unique():
            subject_data = df[df['id'] == subject].sort_values('time')
            if len(subject_data) > 0:
                first_timepoint = subject_data.iloc[0]
                row_data = {'subject_id': subject, 'time': first_timepoint['time']}
                for col in volume_cols:
                    row_data[col] = first_timepoint[col]
                baseline_data.append(row_data)
        baseline_df = pd.DataFrame(baseline_data)
        print(f"Loaded baseline volumes for {len(baseline_df)} subjects")
        return baseline_df
    except Exception as e:
        print(f"Error loading baseline volumes: {e}")
        return None


def load_braingen_predictions(predictions_dir='./predictions'):
    """
    Load BrainGenFlow trajectory predictions and compute per-ROI OLS slopes.

    Each CSV file in predictions_dir is named trajectory_<subject_id>_fold<N>.csv
    and contains 145 ROI pairs of rows:
        <roi>_real  — sparse observed volumes at actual visit months (NaN elsewhere)
        <roi>_gen   — dense generated trajectory across all 70 months

    Returns a DataFrame with columns matching the RNN-AD / DKGP RoC format:
        subject_id, roi, real_slope, pred_slope, num_timepoints,
        time_range, real_initial_value, pred_initial_value, model
    """
    import re
    from pathlib import Path

    predictions_dir = Path(predictions_dir)
    months     = np.arange(70)
    time_years = months / 12.0

    def _ols_slope(t, v):
        if len(t) < 2:
            return np.nan
        return float(np.polyfit(t, v, 1)[0])

    csv_files = sorted(predictions_dir.glob('trajectory_*_fold*.csv'))
    print(f"  Found {len(csv_files)} BrainGenFlow trajectory files in '{predictions_dir}'")

    all_records = []
    for fpath in csv_files:
        m = re.match(r'trajectory_(.+)_fold\d+\.csv', fpath.name)
        if not m:
            continue
        subject_id = m.group(1)

        traj_df = pd.read_csv(fpath, index_col=0)

        for roi_idx in range(145):
            real_key = f'{roi_idx}_real'
            gen_key  = f'{roi_idx}_gen'

            if real_key not in traj_df.index or gen_key not in traj_df.index:
                continue

            real_vals = traj_df.loc[real_key].values.astype(float)
            gen_vals  = traj_df.loc[gen_key].values.astype(float)

            real_mask = ~np.isnan(real_vals)
            gen_mask  = ~np.isnan(gen_vals)

            real_t = time_years[real_mask]
            real_v = real_vals[real_mask]
            gen_t  = time_years[gen_mask]
            gen_v  = gen_vals[gen_mask]

            all_records.append({
                'subject_id':         subject_id,
                'roi':                roi_idx,
                'real_slope':         _ols_slope(real_t, real_v),
                'pred_slope':         _ols_slope(gen_t, gen_v),
                'num_timepoints':     int(real_mask.sum()),
                'time_range':         float(real_t.max() - real_t.min()) if real_mask.sum() >= 2 else np.nan,
                'real_initial_value': float(real_v[0]) if len(real_v) > 0 else np.nan,
                'pred_initial_value': float(gen_v[0]) if len(gen_v) > 0 else np.nan,
                'model':              'BrainGenFlow'
            })

    braingen_df = pd.DataFrame(all_records)
    n_subjects  = braingen_df['subject_id'].nunique() if len(braingen_df) > 0 else 0
    print(f"  BrainGenFlow RoC: {len(all_records):,} measurements, {n_subjects} subjects")
    return braingen_df


def perform_false_discovery_analysis(y_true, y_pred, y_prob, alpha=0.05):
    """Perform false discovery rate analysis."""
    contingency_table = confusion_matrix(y_true, y_pred)
    if contingency_table.shape == (2, 2):
        chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
    else:
        p_value = 1.0

    fdr = 1 - precision_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    _, p_adjusted, _, _ = multipletests([p_value], method='fdr_bh', alpha=alpha)

    return {
        'fdr':         fdr,
        'fpr':         fpr,
        'p_value':     p_value,
        'p_adjusted':  p_adjusted[0],
        'significant': p_adjusted[0] < alpha
    }


def delong_test(y_true, y_prob_a, y_prob_b):
    """
    DeLong's test for comparing two correlated ROC curves.

    Reference: DeLong, DeLong & Clarke-Pearson (1988), Biometrics 44(3):837-845.

    This is the gold standard for comparing AUCs from two classifiers evaluated
    on the same set of subjects. It exploits the full paired structure of the
    predictions — each subject contributes one probability score per model —
    giving far more power than a t-test on 5 fold AUCs.

    Parameters
    ----------
    y_true   : array-like, shape (n,)  — binary ground truth labels
    y_prob_a : array-like, shape (n,)  — predicted probabilities from model A
    y_prob_b : array-like, shape (n,)  — predicted probabilities from model B

    Returns
    -------
    dict with keys: auc_a, auc_b, z_stat, p_value, significant
    """
    y_true   = np.asarray(y_true)
    y_prob_a = np.asarray(y_prob_a)
    y_prob_b = np.asarray(y_prob_b)

    def compute_midrank(x):
        """Compute midranks for the DeLong structural component."""
        J      = np.argsort(x)
        Z      = x[J]
        N      = len(x)
        T      = np.zeros(N, dtype=float)
        i      = 0
        while i < N:
            j = i
            while j < N - 1 and Z[j] == Z[j + 1]:
                j += 1
            T[i:j+1] = 0.5 * (i + j + 2)   # 1-based midrank
            i = j + 1
        T2 = np.empty(N, dtype=float)
        T2[J] = T
        return T2

    def structural_components(y_true, y_prob):
        """
        Compute the structural components V10 and V01 used in DeLong's variance.
        V10[i] = P(score of positive i > score of random negative)  — placement value
        V01[j] = P(score of random positive > score of negative j)  — placement value
        """
        pos_idx = np.where(y_true == 1)[0]
        neg_idx = np.where(y_true == 0)[0]
        m = len(pos_idx)   # number of positives
        n = len(neg_idx)   # number of negatives

        pos_scores = y_prob[pos_idx]
        neg_scores = y_prob[neg_idx]

        # Placement values for positives (V10)
        V10 = np.zeros(m)
        for i, ps in enumerate(pos_scores):
            V10[i] = np.mean(ps > neg_scores) + 0.5 * np.mean(ps == neg_scores)

        # Placement values for negatives (V01)
        V01 = np.zeros(n)
        for j, ns in enumerate(neg_scores):
            V01[j] = np.mean(pos_scores > ns) + 0.5 * np.mean(pos_scores == ns)

        auc = np.mean(V10)
        return auc, V10, V01, m, n

    auc_a, V10_a, V01_a, m, n = structural_components(y_true, y_prob_a)
    auc_b, V10_b, V01_b, _, _ = structural_components(y_true, y_prob_b)

    # Covariance matrix of [AUC_a, AUC_b] using DeLong's variance estimator
    # S = (1/m)*S10 + (1/n)*S01  where S10, S01 are 2x2 covariance matrices
    S10 = np.cov(np.vstack([V10_a, V10_b]))   # shape (2,2)
    S01 = np.cov(np.vstack([V01_a, V01_b]))   # shape (2,2)

    S = S10 / m + S01 / n   # variance of [AUC_a, AUC_b]

    # Variance of the difference AUC_a - AUC_b
    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]

    if var_diff <= 0:
        return {
            'auc_a': auc_a, 'auc_b': auc_b,
            'z_stat': 0.0, 'p_value': 1.0, 'significant': False,
            'note': 'var_diff <= 0 — cannot compute z-statistic'
        }

    z_stat  = (auc_a - auc_b) / np.sqrt(var_diff)
    p_value = 2 * stats.norm.sf(abs(z_stat))   # two-sided

    return {
        'auc_a':       auc_a,
        'auc_b':       auc_b,
        'z_stat':      z_stat,
        'p_value':     p_value,
        'significant': p_value < 0.05
    }


def evaluate_features(X, y, feature_type):
    """
    Evaluate features using a nested cross-validation scheme:

    Outer loop — StratifiedKFold(5):
        Each fold holds out 20% as a clean test set, never seen during training.

    Inner loop — for each outer train split (80%):
        A GridSearchCV with StratifiedKFold(3) tunes hyperparameters on train/val.
        The best estimator is then refit on the full 80% and evaluated on the 20% test.

    This gives 5 held-out test AUCs per classifier. The mean ± std across those
    5 folds is the reported performance, and the 5 fold AUCs are stored for the
    paired t-test in perform_statistical_comparison.

    The mean ROC curve is built by interpolating each fold's TPR onto a common
    FPR grid — the standard publication approach.
    """
    from sklearn.model_selection import StratifiedKFold, GridSearchCV

    print(f"\n{'='*80}")
    print(f"ANALYSIS USING {feature_type.upper()} FEATURES")
    print(f"{'='*80}")
    print(f"Scheme: 5-fold outer CV | inner GridSearchCV(3-fold) for hyperparameter tuning")

    X = np.array(X)
    y = np.array(y)

    # ── Outer CV: 5 stratified folds ────────────────────────────────────────
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── Model search spaces ──────────────────────────────────────────────────
    model_configs = {
        'Logistic Regression': {
            'estimator': LogisticRegression(
                max_iter=2000, class_weight='balanced', random_state=42),
            'param_grid': {
                'C': [0.01, 0.1, 1.0, 10.0]
            }
        },
        'Random Forest': {
            'estimator': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'param_grid': {
                'n_estimators': [100, 200],
                'max_depth':    [None, 5, 10]
            }
        },
        'Gradient Boosting': {
            'estimator': GradientBoostingClassifier(random_state=42),
            'param_grid': {
                'n_estimators':  [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth':     [3, 5]
            }
        }
    }

    # Common FPR grid for mean ROC construction
    mean_fpr = np.linspace(0, 1, 300)

    results = {}
    for name, config in model_configs.items():
        print(f"\n{name}:")
        print("-" * len(name))

        # Per-fold accumulators
        fold_aucs       = []
        fold_pr_aucs    = []
        fold_accuracies = []
        fold_precisions = []
        fold_recalls    = []
        fold_f1s        = []
        fold_fdrs       = []
        fold_fprs_stat  = []
        tprs_interp     = []
        best_params_log = []
        # Store raw predictions for DeLong's test (pooled across folds)
        all_y_true      = []
        all_y_prob      = []

        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
            X_train_outer, X_test = X[train_idx], X[test_idx]
            y_train_outer, y_test = y[train_idx], y[test_idx]

            # Scale: fit only on outer train, apply to both
            scaler         = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_outer)
            X_test_scaled  = scaler.transform(X_test)

            # Inner GridSearchCV on the outer train split
            gs = GridSearchCV(
                estimator  = config['estimator'],
                param_grid = config['param_grid'],
                cv         = inner_cv,
                scoring    = 'roc_auc',
                n_jobs     = -1,
                refit      = True   # refit best estimator on full outer train
            )
            gs.fit(X_train_scaled, y_train_outer)
            best_params_log.append(gs.best_params_)

            # Evaluate best model on held-out test set
            best_model = gs.best_estimator_
            y_pred = best_model.predict(X_test_scaled)
            y_prob = (best_model.predict_proba(X_test_scaled)[:, 1]
                      if hasattr(best_model, "predict_proba")
                      else best_model.decision_function(X_test_scaled))

            # Accumulate metrics
            fold_aucs.append(roc_auc_score(y_test, y_prob))
            fold_pr_aucs.append(average_precision_score(y_test, y_prob))
            fold_accuracies.append(accuracy_score(y_test, y_pred))
            fold_precisions.append(precision_score(y_test, y_pred, zero_division=0))
            fold_recalls.append(recall_score(y_test, y_pred, zero_division=0))
            fold_f1s.append(f1_score(y_test, y_pred, zero_division=0))

            fdr_res = perform_false_discovery_analysis(y_test, y_pred, y_prob)
            fold_fdrs.append(fdr_res['fdr'])
            fold_fprs_stat.append(fdr_res['fpr'])

            # Store raw predictions for DeLong (pooled across folds)
            all_y_true.extend(y_test.tolist())
            all_y_prob.extend(y_prob.tolist())

            # Interpolate TPR onto common FPR grid
            fpr_fold, tpr_fold, _ = roc_curve(y_test, y_prob)
            tpr_i = np.interp(mean_fpr, fpr_fold, tpr_fold)
            tpr_i[0] = 0.0
            tprs_interp.append(tpr_i)

            print(f"  Fold {fold_idx}: AUC = {fold_aucs[-1]:.3f}  "
                  f"best params = {gs.best_params_}")

        # Mean ROC curve across 5 folds
        mean_tpr     = np.mean(tprs_interp, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr      = np.std(tprs_interp, axis=0)

        mean_auc = np.mean(fold_aucs)
        std_auc  = np.std(fold_aucs)

        print(f"\n  → Mean ROC-AUC : {mean_auc:.3f} ± {std_auc:.3f}  [5 held-out test folds]")
        print(f"  → Mean PR-AUC  : {np.mean(fold_pr_aucs):.3f} ± {np.std(fold_pr_aucs):.3f}")
        print(f"  → Mean Accuracy: {np.mean(fold_accuracies):.3f} ± {np.std(fold_accuracies):.3f}")
        print(f"  → Mean Precision: {np.mean(fold_precisions):.3f} ± {np.std(fold_precisions):.3f}")
        print(f"  → Mean Recall  : {np.mean(fold_recalls):.3f} ± {np.std(fold_recalls):.3f}")
        print(f"  → Mean F1      : {np.mean(fold_f1s):.3f} ± {np.std(fold_f1s):.3f}")
        print(f"\n  False Discovery Analysis (mean across folds):")
        print(f"    FDR: {np.mean(fold_fdrs):.3f} ± {np.std(fold_fdrs):.3f}")
        print(f"    FPR: {np.mean(fold_fprs_stat):.3f} ± {np.std(fold_fprs_stat):.3f}")

        results[name] = {
            'roc_auc':      mean_auc,
            'roc_auc_std':  std_auc,
            'pr_auc':       np.mean(fold_pr_aucs),
            'pr_auc_std':   np.std(fold_pr_aucs),
            'accuracy':     np.mean(fold_accuracies),
            'precision':    np.mean(fold_precisions),
            'recall':       np.mean(fold_recalls),
            'f1':           np.mean(fold_f1s),
            'fdr':          np.mean(fold_fdrs),
            'fpr_stat':     np.mean(fold_fprs_stat),
            'fold_aucs':    np.array(fold_aucs),    # 5 values for paired t-test
            'fpr':          mean_fpr,               # mean ROC x-axis
            'tpr':          mean_tpr,               # mean ROC y-axis
            'tpr_std':      std_tpr,                # ± band
            'best_params':  best_params_log,        # hyperparams per fold
            'all_y_true':   np.array(all_y_true),   # pooled labels  — for DeLong
            'all_y_prob':   np.array(all_y_prob),   # pooled probs   — for DeLong
        }

    return results


def compute_95ci(fold_aucs):
    """
    Compute 95% CI for mean AUC from k-fold CV results using
    the t-distribution with df = k - 1.

    Parameters
    ----------
    fold_aucs : array-like, shape (k,) — per-fold AUC values

    Returns
    -------
    mean, ci_lower, ci_upper
    """
    from scipy import stats
    fold_aucs = np.asarray(fold_aucs)
    k         = len(fold_aucs)
    mean      = np.mean(fold_aucs)
    se        = stats.sem(fold_aucs)          # std / sqrt(k)
    t_crit    = stats.t.ppf(0.975, df=k - 1) # two-sided, df = k-1 = 4
    margin    = t_crit * se
    return mean, mean - margin, mean + margin

def perform_statistical_comparison(baseline_results, rnn_results, dkgp_results,
                                    real_results,
                                    braingen_results=None,
                                    alpha=0.05):
    """
    Statistical comparison across feature sets using DeLong's test on pooled CV
    predictions (n ≈ 882).

    Pairwise comparisons (symmetrically applied classifier):
      1. BrainGenFlow predicted vs. Baseline volumes  (primary claim)
      2. BrainGenFlow predicted vs. RNN-AD predicted  (head-to-head)
      3. BrainGenFlow predicted vs. DKGP predicted    (head-to-head)
      4. DKGP predicted         vs. Baseline volumes
      5. RNN-AD predicted       vs. Baseline volumes
      6. DKGP predicted         vs. RNN-AD predicted

    DeLong's test requires row-level alignment of pooled prediction vectors —
    guaranteed by StratifiedKFold(random_state=42) and canonical subject ordering.
    """
    print("\n" + "="*80)
    print("DELONG'S TEST — ALL PAIRWISE COMPARISONS")
    print(f"Classifier : {DKGP_CLASSIFIER} (applied symmetrically to all feature sets)")
    ref_n = len(dkgp_results[DKGP_CLASSIFIER]['all_y_true'])
    print(f"n          ≈ {ref_n} subjects (pooled CV)")
    print("="*80)

    # ── Extract pooled prediction vectors ─────────────────────────────────────
    y_true_base = baseline_results[FIXED_CLASSIFIER]['all_y_true']
    y_true_rnn  = rnn_results[RNN_CLASSIFIER]['all_y_true']
    y_true_dkgp = dkgp_results[DKGP_CLASSIFIER]['all_y_true']

    y_prob_base = baseline_results[RNN_CLASSIFIER]['all_y_prob']
    y_prob_rnn  = rnn_results[RNN_CLASSIFIER]['all_y_prob']
    y_prob_dkgp = dkgp_results[DKGP_CLASSIFIER]['all_y_prob']

    # ── Alignment assertion ───────────────────────────────────────────────────
    assert np.array_equal(y_true_base, y_true_rnn) and \
           np.array_equal(y_true_rnn,  y_true_dkgp), \
        "y_true vectors differ across feature sets — CV splits are misaligned. " \
        "Ensure all calls to evaluate_features use random_state=42 and receive " \
        "subjects in the same order."

    y_true = y_true_dkgp   # all identical; use any one

    # ── BrainGenFlow predictions (optional) ──────────────────────────────────
    y_prob_braingen = None
    if braingen_results is not None:
        y_true_braingen = braingen_results[BRAINGEN_CLASSIFIER]['all_y_true']
        assert np.array_equal(y_true_dkgp, y_true_braingen), \
            "y_true vectors differ: BrainGenFlow vs DKGP — subject ordering misaligned."
        y_prob_braingen = braingen_results[BRAINGEN_CLASSIFIER]['all_y_prob']

    # ── AUC summary ──────────────────────────────────────────────────────────
    auc_base = baseline_results[FIXED_CLASSIFIER]['roc_auc']
    auc_rnn  = rnn_results[RNN_CLASSIFIER]['roc_auc']
    auc_dkgp = dkgp_results[DKGP_CLASSIFIER]['roc_auc']
    auc_real = real_results[FIXED_CLASSIFIER]['roc_auc']

    std_base = baseline_results[FIXED_CLASSIFIER]['roc_auc_std']
    std_rnn  = rnn_results[RNN_CLASSIFIER]['roc_auc_std']
    std_dkgp = dkgp_results[DKGP_CLASSIFIER]['roc_auc_std']
    std_real = real_results[FIXED_CLASSIFIER]['roc_auc_std']

    # ── AUC summary with 95% CI ───────────────────────────────────────────────
    print(f"\n  AUC Summary (mean [95% CI] across 5 folds):")

    summary_rows = [
        ("Real upper bound",        real_results,      FIXED_CLASSIFIER),
        ("BrainGenFlow predicted",  braingen_results,  BRAINGEN_CLASSIFIER),
        ("DKGP predicted",          dkgp_results,      DKGP_CLASSIFIER),
        ("RNN-AD predicted",        rnn_results,       RNN_CLASSIFIER),
        ("Baseline volumes",        baseline_results,  FIXED_CLASSIFIER),
    ]
    for label, results, classifier in summary_rows:
        if results is None:
            continue
        fold_aucs          = results[classifier]['fold_aucs']
        mean, lower, upper = compute_95ci(fold_aucs)
        print(f"    {label:<28}: {mean:.4f} [95% CI: {lower:.4f}, {upper:.4f}]")

    # ── Pairwise DeLong tests ─────────────────────────────────────────────────
    comparisons = [
        ("DKGP predicted",   "Baseline volumes",  y_prob_dkgp, y_prob_base),
        ("RNN-AD predicted", "Baseline volumes",  y_prob_rnn,  y_prob_base),
        ("DKGP predicted",   "RNN-AD predicted",  y_prob_dkgp, y_prob_rnn),
    ]
    if y_prob_braingen is not None:
        comparisons = [
            ("BrainGenFlow predicted", "Baseline volumes",  y_prob_braingen, y_prob_base),
            ("BrainGenFlow predicted", "RNN-AD predicted",  y_prob_braingen, y_prob_rnn),
            ("BrainGenFlow predicted", "DKGP predicted",    y_prob_braingen, y_prob_dkgp),
            ("DKGP predicted",         "Baseline volumes",  y_prob_dkgp,     y_prob_base),
            ("RNN-AD predicted",       "Baseline volumes",  y_prob_rnn,      y_prob_base),
            ("DKGP predicted",         "RNN-AD predicted",  y_prob_dkgp,     y_prob_rnn),
        ]

    delong_results = {}
    for name_a, name_b, prob_a, prob_b in comparisons:
        key = f"{name_a} vs {name_b}"
        print(f"\n{'─'*60}")
        print(f"  {key}")
        print(f"{'─'*60}")

        res = delong_test(y_true, prob_a, prob_b)
        delong_results[key] = res

        delta = res['auc_a'] - res['auc_b']
        print(f"  AUC ({name_a:<28}): {res['auc_a']:.4f}")
        print(f"  AUC ({name_b:<28}): {res['auc_b']:.4f}")
        print(f"  ΔAUC                          : {delta:+.4f}")
        print(f"  z-statistic                   : {res['z_stat']:.3f}")
        print(f"  p-value                       : {res['p_value']:.4e}")
        print(f"  Significant (α = {alpha})      : {'✓ Yes' if res['significant'] else '✗ No'}")
        if 'note' in res:
            print(f"  Note: {res['note']}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("SUMMARY TABLE")
    print(f"{'─'*60}")
    header = f"  {'Comparison':<46} {'ΔAUC':>8}  {'z':>7}  {'p':>12}  {'Sig':>5}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key, res in delong_results.items():
        delta = res['auc_a'] - res['auc_b']
        sig   = '✓' if res['significant'] else '✗'
        print(f"  {key:<46} {delta:>+8.4f}  {res['z_stat']:>7.3f}  "
              f"{res['p_value']:>12.4e}  {sig:>5}")

    # ── Manuscript-ready conclusion ───────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("MANUSCRIPT CONCLUSION")
    print(f"{'─'*60}")

    conclusion_pairs = []
    if y_prob_braingen is not None:
        conclusion_pairs += [
            ("BrainGenFlow vs Baseline", delong_results["BrainGenFlow predicted vs Baseline volumes"], "BrainGenFlow", "Baseline"),
            ("BrainGenFlow vs RNN-AD",   delong_results["BrainGenFlow predicted vs RNN-AD predicted"], "BrainGenFlow", "RNN-AD"),
            ("BrainGenFlow vs DKGP",     delong_results["BrainGenFlow predicted vs DKGP predicted"],   "BrainGenFlow", "DKGP"),
        ]
    conclusion_pairs += [
        ("DKGP vs Baseline", delong_results["DKGP predicted vs Baseline volumes"],  "DKGP",   "Baseline"),
        ("RNN vs Baseline",  delong_results["RNN-AD predicted vs Baseline volumes"], "RNN-AD", "Baseline"),
        ("DKGP vs RNN-AD",   delong_results["DKGP predicted vs RNN-AD predicted"],  "DKGP",   "RNN-AD"),
    ]

    for label, res, name_a, name_b in conclusion_pairs:
        sig_str = "significantly outperforms" if res['significant'] else \
                  "does not significantly outperform"
        print(f"  {label}: {name_a} {sig_str} {name_b} "
              f"(ΔAUC = {res['auc_a'] - res['auc_b']:+.4f}, "
              f"z = {res['z_stat']:.3f}, p = {res['p_value']:.4e})")

    out = {
        'delong_dkgp_vs_baseline': delong_results["DKGP predicted vs Baseline volumes"],
        'delong_rnn_vs_baseline':  delong_results["RNN-AD predicted vs Baseline volumes"],
        'delong_dkgp_vs_rnn':      delong_results["DKGP predicted vs RNN-AD predicted"],
        'auc_summary': {
            'baseline': (auc_base, std_base),
            'rnn':      (auc_rnn,  std_rnn),
            'dkgp':     (auc_dkgp, std_dkgp),
            'real':     (auc_real, std_real),
        }
    }
    if y_prob_braingen is not None:
        out['delong_braingen_vs_baseline'] = delong_results["BrainGenFlow predicted vs Baseline volumes"]
        out['delong_braingen_vs_rnn']      = delong_results["BrainGenFlow predicted vs RNN-AD predicted"]
        out['delong_braingen_vs_dkgp']     = delong_results["BrainGenFlow predicted vs DKGP predicted"]
        out['auc_summary']['braingen']     = (braingen_results[BRAINGEN_CLASSIFIER]['roc_auc'],
                                               braingen_results[BRAINGEN_CLASSIFIER]['roc_auc_std'])
    return out

def analyze_slope_quality(rate_of_change_df):
    """Analyze the quality of predicted slopes compared to real slopes."""
    print(f"\n{'='*80}")
    print("SLOPE QUALITY ANALYSIS")
    print(f"{'='*80}")

    valid_data = rate_of_change_df.dropna(subset=['real_slope', 'pred_slope'])
    if len(valid_data) == 0:
        print("No valid slope data found for analysis")
        return None

    real_stats = valid_data['real_slope'].describe()
    pred_stats = valid_data['pred_slope'].describe()

    print("\nReal Slope Statistics:")
    for k in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
        print(f"  {k}: {real_stats[k]:.3f}")

    print("\nPredicted Slope Statistics:")
    for k in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
        print(f"  {k}: {pred_stats[k]:.3f}")

    correlation    = valid_data['real_slope'].corr(valid_data['pred_slope'])
    r_squared      = correlation ** 2
    mae            = np.mean(np.abs(valid_data['real_slope'] - valid_data['pred_slope']))
    rmse           = np.sqrt(np.mean((valid_data['real_slope'] - valid_data['pred_slope'])**2))
    variance_ratio = (pred_stats['std'] / real_stats['std'])**2

    print(f"\nCorrelation between real and predicted slopes: {correlation:.3f}")
    print(f"R-squared: {r_squared:.3f}")
    print(f"Mean Absolute Error: {mae:.3f}")
    print(f"Root Mean Squared Error: {rmse:.3f}")
    print(f"Variance Ratio (Predicted/Real): {variance_ratio:.3f}")

    for col, label in [('num_timepoints', 'Number of Timepoints'),
                       ('time_range',     'Time Range (years)')]:
        s = valid_data[col].describe()
        print(f"\n{label} Statistics:")
        for k in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
            print(f"  {k}: {s[k]:.1f}")

    print(f"\n{'='*80}")
    print("SLOPE QUALITY ASSESSMENT")
    print(f"{'='*80}")

    if   correlation > 0.8: corr_quality = "Excellent"
    elif correlation > 0.6: corr_quality = "Good"
    elif correlation > 0.4: corr_quality = "Moderate"
    elif correlation > 0.2: corr_quality = "Poor"
    else:                   corr_quality = "Very Poor"
    print(f"Correlation Quality: {corr_quality} ({correlation:.3f})")

    if   0.8 < variance_ratio < 1.2: var_quality = "Good (variance preserved)"
    elif 0.5 < variance_ratio < 2.0: var_quality = "Moderate (some variance distortion)"
    else:                            var_quality = "Poor (significant variance distortion)"
    print(f"Variance Quality: {var_quality} (ratio: {variance_ratio:.3f})")

    if   mae < 0.01: error_quality = "Excellent"
    elif mae < 0.02: error_quality = "Good"
    elif mae < 0.05: error_quality = "Moderate"
    else:            error_quality = "Poor"
    print(f"Error Quality: {error_quality} (MAE: {mae:.3f})")

    print(f"\nOverall Slope Quality Assessment:")
    if correlation > 0.6 and 0.5 < variance_ratio < 2.0 and mae < 0.02:
        overall_quality = "Good"
        print(f"  The model shows good ability to predict rate of change")
    elif correlation > 0.4 and 0.3 < variance_ratio < 3.0 and mae < 0.05:
        overall_quality = "Moderate"
        print(f"  The model shows moderate ability to predict rate of change")
    else:
        overall_quality = "Poor"
        print(f"  The model shows limited ability to predict rate of change")
    print(f"  Overall Quality: {overall_quality}")

    plt.figure(figsize=(10, 6))
    plt.hist(valid_data['real_slope'], bins=50, alpha=0.5, label='Real Slopes',      density=True)
    plt.hist(valid_data['pred_slope'], bins=50, alpha=0.5, label='Predicted Slopes', density=True)
    plt.xlabel('Slope Value', fontweight='bold')
    plt.ylabel('Density',     fontweight='bold')
    plt.title('Distribution of Real vs Predicted Slopes', pad=20, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./mciprogression/RNNAD_DKGP_slope_quality_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('./mciprogression/RNNAD_DKGP_slope_quality_comparison.svg',              bbox_inches='tight')
    plt.close()
    print(f"\nSlope quality comparison plot saved.")

    return {
        'correlation':     correlation,
        'r_squared':       r_squared,
        'mae':             mae,
        'rmse':            rmse,
        'variance_ratio':  variance_ratio,
        'overall_quality': overall_quality
    }


def analyze_progressor_classification(rate_of_change_df, covariates_df, baseline_df=None):
    """Analyze progressor vs non-progressor classification using real and/or predicted data."""

    with open('./manuscript1/RNNAD_DKGP_rate_of_change_mci_stable_mci_progressor.txt', 'w') as f:
        import sys
        original_stdout = sys.stdout

        class TeeOutput:
            def __init__(self, file):
                self.file   = file
                self.stdout = sys.stdout
            def write(self, text):
                self.stdout.write(text)
                self.file.write(text)
            def flush(self):
                self.stdout.flush()
                self.file.flush()

        sys.stdout = TeeOutput(f)

        try:
            print("\n" + "="*80)
            print("DATASET INFORMATION")
            print("="*80)

            # ── Separate by model ────────────────────────────────────────────
            rnn_data       = rate_of_change_df[rate_of_change_df['model'] == 'RNN-AD']
            dkgp_data      = rate_of_change_df[rate_of_change_df['model'] == 'DKGP']
            braingen_data  = rate_of_change_df[rate_of_change_df['model'] == 'BrainGenFlow']
            dkgp_data      = dkgp_data.dropna(subset=['real_slope', 'pred_slope'])
            braingen_data  = braingen_data.dropna(subset=['real_slope', 'pred_slope'])

            # ── ROI coverage check ───────────────────────────────────────────
            dkgp_roi_counts      = dkgp_data.groupby('subject_id')['roi'].nunique()
            rnn_roi_counts       = rnn_data.groupby('subject_id')['roi'].nunique()
            braingen_roi_counts  = braingen_data.groupby('subject_id')['roi'].nunique()
            print(f"DKGP subjects with < 145 ROIs: {(dkgp_roi_counts < 145).sum()}")
            print(f"RNN-AD subjects with < 145 ROIs: {(rnn_roi_counts < 145).sum()}")
            print(f"BrainGenFlow subjects with < 145 ROIs: {(braingen_roi_counts < 145).sum()}")
            print(f"DKGP ROI count distribution:\n{dkgp_roi_counts.value_counts().sort_index()}")

            # ── NaN check ────────────────────────────────────────────────────
            for label, df in [('RNN-AD', rnn_data), ('DKGP', dkgp_data), ('BrainGenFlow', braingen_data)]:
                n_nan_real  = df['real_slope'].isna().sum()
                n_nan_pred  = df['pred_slope'].isna().sum()
                n_subj_real = df[df['real_slope'].isna()]['subject_id'].nunique()
                n_subj_pred = df[df['pred_slope'].isna()]['subject_id'].nunique()
                print(f"\n[{label}] NaN check:")
                print(f"  real_slope NaNs: {n_nan_real} rows across {n_subj_real} subjects")
                print(f"  pred_slope NaNs: {n_nan_pred} rows across {n_subj_pred} subjects")
                print(f"  NaN ROIs (real): {sorted(df[df['real_slope'].isna()]['roi'].unique().tolist())}")
                print(f"  NaN ROIs (pred): {sorted(df[df['pred_slope'].isna()]['roi'].unique().tolist())}")

            print(f"\nRNN-AD data: {len(rnn_data)} rows")
            print(f"DKGP data: {len(dkgp_data)} rows")
            print(f"BrainGenFlow data: {len(braingen_data)} rows")
            print(f"Number of ROIs: {len(rate_of_change_df['roi'].unique())}")

            # ── Pivot predicted slopes ───────────────────────────────────────
            rnn_pred_wide = rnn_data.pivot(
                index='subject_id', columns='roi', values='pred_slope'
            ).reset_index()
            rnn_pred_wide.columns = ['subject_id'] + [
                f'pred_roc_{col}' for col in rnn_pred_wide.columns[1:]
            ]

            dkgp_pred_wide = dkgp_data.pivot(
                index='subject_id', columns='roi', values='pred_slope'
            ).reset_index()
            dkgp_pred_wide.columns = ['subject_id'] + [
                f'pred_roc_{col}' for col in dkgp_pred_wide.columns[1:]
            ]

            braingen_pred_wide = braingen_data.pivot(
                index='subject_id', columns='roi', values='pred_slope'
            ).reset_index()
            braingen_pred_wide.columns = ['subject_id'] + [
                f'pred_roc_{col}' for col in braingen_pred_wide.columns[1:]
            ]

            # Real upper bound comes from RNN-AD file — same preprocessing
            # guarantees real AUC >= predicted AUC for RNN-AD
            real_wide = rnn_data.pivot(
                index='subject_id', columns='roi', values='real_slope'
            ).reset_index()
            real_wide.columns = ['subject_id'] + [
                f'real_roc_{col}' for col in real_wide.columns[1:]
            ]

            print(f"\nRNN-AD predicted subjects:     {len(rnn_pred_wide)}")
            print(f"DKGP predicted subjects:       {len(dkgp_pred_wide)}")
            print(f"BrainGenFlow predicted subjects:{len(braingen_pred_wide)}")
            print(f"Real upper-bound subjects:      {len(real_wide)}")

            # ── Intersect subjects across slope feature sets ──────────────────
            common_subjects = (
                set(rnn_pred_wide['subject_id'])
                & set(dkgp_pred_wide['subject_id'])
                & set(real_wide['subject_id'])
                & set(braingen_pred_wide['subject_id'])
            )
            print(f"\nCommon subjects across all slope feature sets: {len(common_subjects)}")

            rnn_pred_wide      = rnn_pred_wide[rnn_pred_wide['subject_id'].isin(common_subjects)]
            dkgp_pred_wide     = dkgp_pred_wide[dkgp_pred_wide['subject_id'].isin(common_subjects)]
            braingen_pred_wide = braingen_pred_wide[braingen_pred_wide['subject_id'].isin(common_subjects)]
            real_wide          = real_wide[real_wide['subject_id'].isin(common_subjects)]

            # ── Slope quality ────────────────────────────────────────────────
            print("\n" + "="*80)
            print("SLOPE QUALITY ANALYSIS - RNN-AD MODEL")
            print("="*80)
            rnn_slope_quality = analyze_slope_quality(
                rate_of_change_df[rate_of_change_df['model'] == 'RNN-AD']
            )

            print("\n" + "="*80)
            print("SLOPE QUALITY ANALYSIS - DKGP MODEL")
            print("="*80)
            dkgp_slope_quality = analyze_slope_quality(
                rate_of_change_df[rate_of_change_df['model'] == 'DKGP']
            )

            if rnn_slope_quality and dkgp_slope_quality:
                print("\n" + "="*80)
                print("SLOPE QUALITY COMPARISON")
                print("="*80)
                print(f"  RNN-AD Correlation: {rnn_slope_quality['correlation']:.3f}  |  Overall: {rnn_slope_quality['overall_quality']}")
                print(f"  DKGP   Correlation: {dkgp_slope_quality['correlation']:.3f}  |  Overall: {dkgp_slope_quality['overall_quality']}")
                print(f"  Difference (DKGP - RNN-AD): {dkgp_slope_quality['correlation'] - rnn_slope_quality['correlation']:.3f}")
                better = "DKGP" if dkgp_slope_quality['correlation'] > rnn_slope_quality['correlation'] else "RNN-AD"
                print(f"\n  {better} shows better slope prediction quality")

            # ── Build progression labels ─────────────────────────────────────
            progression_data = []
            for subject in covariates_df['PTID'].unique():
                subject_data = covariates_df[covariates_df['PTID'] == subject].sort_values('Time')
                if len(subject_data) > 1:
                    initial_dx = subject_data.iloc[0]['Diagnosis']
                    final_dx   = subject_data.iloc[-1]['Diagnosis']
                    if isinstance(initial_dx, (int, float, np.int64, np.float64)) and \
                       isinstance(final_dx,   (int, float, np.int64, np.float64)):
                        if initial_dx == 1:
                            progression_data.append({
                                'PTID':              subject,
                                'initial_diagnosis': initial_dx,
                                'final_diagnosis':   final_dx,
                                'is_progressor':     int(initial_dx == 1 and final_dx == 2)
                            })

            progression_df   = pd.DataFrame(progression_data)
            overlap_subjects = set(progression_df['PTID']) & common_subjects
            progression_df   = progression_df[progression_df['PTID'].isin(overlap_subjects)]

            print(f"\nSubjects with progression data (starting as MCI): {len(set(progression_df['PTID']))}")
            print(f"Subjects with both progression and ROC data:       {len(overlap_subjects)}")
            print(f"\nFinal analysis dataset:")
            print(f"  Total:           {len(progression_df)}")
            print(f"  Progressors:     {progression_df['is_progressor'].sum()}")
            print(f"  Non-Progressors: {(progression_df['is_progressor'] == 0).sum()}")
            print(f"  Progression rate: {progression_df['is_progressor'].mean()*100:.1f}%")

            # ── Add PTID & merge labels ──────────────────────────────────────
            for df in [rnn_pred_wide, dkgp_pred_wide, braingen_pred_wide, real_wide]:
                df['PTID'] = df['subject_id']

            labels = progression_df[['PTID', 'is_progressor']]

            merged_real           = pd.merge(real_wide,           labels, on='PTID', how='inner')
            merged_pred_rnn       = pd.merge(rnn_pred_wide,       labels, on='PTID', how='inner')
            merged_pred_dkgp      = pd.merge(dkgp_pred_wide,      labels, on='PTID', how='inner')
            merged_pred_braingen  = pd.merge(braingen_pred_wide,  labels, on='PTID', how='inner')

            # ── Baseline: intersect with slope cohort ────────────────────────
            merged_baseline  = None
            baseline_results = None
            if baseline_df is not None:
                baseline_df['PTID'] = baseline_df['subject_id']
                baseline_df_filtered = baseline_df[
                    baseline_df['PTID'].isin(set(merged_real['PTID']))
                ]
                merged_baseline = pd.merge(baseline_df_filtered, labels, on='PTID', how='inner')
                print(f"\n  Baseline subjects (after cohort intersection): {len(merged_baseline)}")
            else:
                print("\n  Note: Baseline analysis skipped — baseline volumes not available")

            # ── Canonical subject ordering ────────────────────────────────────
            # All feature sets must contain the same subjects in the same row
            # order so that StratifiedKFold(random_state=42) produces identical
            # fold assignments — a hard requirement for DeLong's test alignment.
            common_ptids = (
                set(merged_real['PTID'])
                & set(merged_pred_rnn['PTID'])
                & set(merged_pred_dkgp['PTID'])
                & set(merged_pred_braingen['PTID'])
                & (set(merged_baseline['PTID']) if merged_baseline is not None
                   else set(merged_real['PTID']))
            )
            print(f"\nSubjects in all feature sets (final intersection): {len(common_ptids)}")

            merged_real          = (merged_real[merged_real['PTID'].isin(common_ptids)]
                                    .sort_values('PTID').reset_index(drop=True))
            merged_pred_rnn      = (merged_pred_rnn[merged_pred_rnn['PTID'].isin(common_ptids)]
                                    .sort_values('PTID').reset_index(drop=True))
            merged_pred_dkgp     = (merged_pred_dkgp[merged_pred_dkgp['PTID'].isin(common_ptids)]
                                    .sort_values('PTID').reset_index(drop=True))
            merged_pred_braingen = (merged_pred_braingen[merged_pred_braingen['PTID'].isin(common_ptids)]
                                    .sort_values('PTID').reset_index(drop=True))
            if merged_baseline is not None:
                merged_baseline  = (merged_baseline[merged_baseline['PTID'].isin(common_ptids)]
                                    .sort_values('PTID').reset_index(drop=True))

            # ── Label alignment verification ──────────────────────────────────
            assert (merged_real['is_progressor'].values ==
                    merged_pred_rnn['is_progressor'].values).all(), \
                "Label mismatch: real vs RNN-AD after sorting."
            assert (merged_real['is_progressor'].values ==
                    merged_pred_dkgp['is_progressor'].values).all(), \
                "Label mismatch: real vs DKGP after sorting."
            assert (merged_real['is_progressor'].values ==
                    merged_pred_braingen['is_progressor'].values).all(), \
                "Label mismatch: real vs BrainGenFlow after sorting."
            if merged_baseline is not None:
                assert (merged_real['is_progressor'].values ==
                        merged_baseline['is_progressor'].values).all(), \
                    "Label mismatch: real vs baseline after sorting."
            print(f"Label alignment verified across all feature sets ✓")
            print(f"Final cohort: {len(merged_real)} subjects  "
                  f"({merged_real['is_progressor'].sum()} progressors, "
                  f"{(merged_real['is_progressor']==0).sum()} non-progressors)")

            # # ── evaluate_features ────────────────────────────────────────────
            # # Classifier is fixed to Logistic Regression across all feature sets
            # # for symmetric DeLong comparisons.
            # CLASSIFIER = 'Logistic Regression'

            if merged_baseline is not None:
                baseline_results = evaluate_features(
                    merged_baseline.drop(
                        ['subject_id', 'PTID', 'is_progressor', 'time'],
                        axis=1, errors='ignore'
                    ),
                    merged_baseline['is_progressor'],
                    'Baseline'
                )

            real_results = evaluate_features(
                merged_real.drop(['subject_id', 'PTID', 'is_progressor'], axis=1),
                merged_real['is_progressor'],
                'Real (upper bound)'
            )

            pred_rnn_results = evaluate_features(
                merged_pred_rnn.drop(['subject_id', 'PTID', 'is_progressor'], axis=1),
                merged_pred_rnn['is_progressor'],
                'RNN-AD Predicted'
            )

            pred_dkgp_results = evaluate_features(
                merged_pred_dkgp.drop(['subject_id', 'PTID', 'is_progressor'], axis=1),
                merged_pred_dkgp['is_progressor'],
                'DKGP Predicted'
            )

            pred_braingen_results = evaluate_features(
                merged_pred_braingen.drop(['subject_id', 'PTID', 'is_progressor'], axis=1),
                merged_pred_braingen['is_progressor'],
                'BrainGenFlow Predicted'
            )

            # ── Cache results ────────────────────────────────────────────────
            CACHE_PATH = './manuscript1/mci_classification_results_cache.pkl'
            os.makedirs('./manuscript1', exist_ok=True)
            with open(CACHE_PATH, 'wb') as f:
                pickle.dump({
                    'baseline_results':      baseline_results,
                    'pred_rnn_results':      pred_rnn_results,
                    'pred_dkgp_results':     pred_dkgp_results,
                    'pred_braingen_results': pred_braingen_results,
                    'real_results':          real_results,
                }, f)
            print(f"\n✓ Results cached to {CACHE_PATH}")

            # ── ROC plot ─────────────────────────────────────────────────────
            plot_comprehensive_roc_curves(
                baseline_results,
                pred_rnn_results,
                pred_dkgp_results,
                real_results,
                save_prefix='mci_stable_vs_progressor',
                braingen_pred_results=pred_braingen_results
            )

            # ── DeLong pairwise comparisons ──────────────────────────────────
            stats_results = perform_statistical_comparison(
                baseline_results  = baseline_results,
                rnn_results       = pred_rnn_results,
                dkgp_results      = pred_dkgp_results,
                real_results      = real_results,
                braingen_results  = pred_braingen_results,
                alpha             = 0.05
            )

            # ── Comprehensive metrics table ───────────────────────────────────
            print(f"\n" + "="*80)
            print("COMPREHENSIVE METRICS SUMMARY (mean ± std across 5 folds)")
            print(f"Classifier: {DKGP_CLASSIFIER} — applied symmetrically to all feature sets")
            print("="*80)

            method_list = [
                ("Real upper bound",        real_results[FIXED_CLASSIFIER]),
                ("BrainGenFlow predicted",  pred_braingen_results[BRAINGEN_CLASSIFIER]),
                ("DKGP predicted",          pred_dkgp_results[DKGP_CLASSIFIER]),
                ("RNN-AD predicted",        pred_rnn_results[RNN_CLASSIFIER]),
            ]
            if baseline_results is not None:
                method_list.append(("Baseline volumes", baseline_results[RNN_CLASSIFIER]))

            metrics = [
                ('roc_auc',   'AUC-ROC'),
                ('pr_auc',    'AUC-PR'),
                ('accuracy',  'Accuracy'),
                ('precision', 'Precision'),
                ('recall',    'Recall'),
                ('f1',        'F1 Score'),
                ('fdr',       'FDR'),
                ('fpr_stat',  'FPR'),
            ]

            col_w = 24
            print(f"\n  {'Metric':<14}", end="")
            for name, _ in method_list:
                print(f"  {name:<{col_w}}", end="")
            print()
            print("  " + "-" * (14 + (col_w + 2) * len(method_list)))

            for key, label in metrics:
                print(f"  {label:<14}", end="")
                for _, r in method_list:
                    val     = r.get(key, float('nan'))
                    std_key = f'{key}_std'
                    cell    = (f"{val:.3f}±{r[std_key]:.3f}"
                               if std_key in r else f"{val:.3f}")
                    print(f"  {cell:<{col_w}}", end="")
                print()

            # ── Delta table: BrainGenFlow vs all others ───────────────────────
            braingen_r = pred_braingen_results[BRAINGEN_CLASSIFIER]
            dkgp_r     = pred_dkgp_results[DKGP_CLASSIFIER]
            rnn_r      = pred_rnn_results[RNN_CLASSIFIER]

            print(f"\n  Delta (BrainGenFlow predicted − RNN-AD predicted):")
            for key, label in metrics:
                bv = braingen_r.get(key, float('nan'))
                rv = rnn_r.get(key, float('nan'))
                if not (np.isnan(bv) or np.isnan(rv)):
                    print(f"    {label:<14}: {bv - rv:+.3f}  "
                          f"{'↑' if bv > rv else '↓'}")

            print(f"\n  Delta (BrainGenFlow predicted − DKGP predicted):")
            for key, label in metrics:
                bv = braingen_r.get(key, float('nan'))
                dv = dkgp_r.get(key, float('nan'))
                if not (np.isnan(bv) or np.isnan(dv)):
                    print(f"    {label:<14}: {bv - dv:+.3f}  "
                          f"{'↑' if bv > dv else '↓'}")

            if baseline_results is not None:
                base_r = baseline_results[RNN_CLASSIFIER]
                print(f"\n  Delta (BrainGenFlow predicted − Baseline volumes):")
                for key, label in metrics:
                    bv = braingen_r.get(key, float('nan'))
                    bsv = base_r.get(key, float('nan'))
                    if not (np.isnan(bv) or np.isnan(bsv)):
                        print(f"    {label:<14}: {bv - bsv:+.3f}  "
                              f"{'↑' if bv > bsv else '↓'}")

            print("\n" + "="*80)
            print("ANALYSIS COMPLETE")
            print("="*80)

        finally:
            sys.stdout = original_stdout

# =============================================================================
# BRAINGEN-ONLY HELPERS
# =============================================================================

def _clean_dx_label(dx) -> str:
    if not isinstance(dx, str):
        return "Other"
    d = dx.strip().lower()
    if "mci" in d or "mild cognitive" in d:
        return "MCI"
    if "ad" in d or "alzheimer" in d or "dementia" in d:
        return "AD"
    if d in ("cn", "cognitively normal", "cognitively unimpaired",
             "cn assumed by study criteria", "normal cognition",
             "normal", "memory complainer (healthy control)"):
        return "CN"
    return "Other"


def build_progression_labels(cov_df: pd.DataFrame) -> pd.DataFrame:
    """
    From longitudinal covariates, identify MCI subjects and assign binary labels:
        1 = progressor   (first dx MCI, last dx AD)
        0 = non-progressor (first dx MCI, last dx MCI)
    Subjects whose final diagnosis is neither MCI nor AD are excluded.
    """
    rows = []
    for ptid, grp in cov_df.groupby("PTID"):
        grp    = grp.sort_values("Time")
        dx_seq = [_clean_dx_label(d) for d in grp["Diagnosis"].tolist()]
        if dx_seq[0] != "MCI":
            continue
        final = dx_seq[-1]
        if final == "AD":
            is_prog = 1
        elif final == "MCI":
            is_prog = 0
        else:
            continue
        rows.append({"PTID": ptid, "is_progressor": is_prog})
    return pd.DataFrame(rows)


def compute_features_from_wide(wide_df: pd.DataFrame):
    """
    Derive three feature matrices from merged_predictions_wide.csv:

    Returns
    -------
    baseline_df   : subjects × 145  — y_ volumes at time=0
    real_roc_df   : subjects × 145  — OLS slope of y_ vs time (in years)
    pred_roc_df   : subjects × 145  — OLS slope of score_ vs time (in years)
    slope_qual_df : long format with (subject_id, roi, real_slope, pred_slope,
                    num_timepoints, time_range, model)  — for analyze_slope_quality
    """
    y_cols     = sorted([c for c in wide_df.columns if c.startswith("y_H_MUSE_Volume_")],
                        key=lambda c: int(c.split("_")[-1]))
    score_cols = sorted([c for c in wide_df.columns if c.startswith("score_H_MUSE_Volume_")],
                        key=lambda c: int(c.split("_")[-1]))

    # A. Baseline: y_ columns at time = 0
    baseline_df = (
        wide_df[wide_df["time"] == 0.0][["id"] + y_cols]
        .rename(columns={"id": "subject_id"})
        .dropna(subset=y_cols)
        .reset_index(drop=True)
    )

    # B+C. Per-subject OLS slopes (months → years) + quality
    real_rows, pred_rows, qual_rows = [], [], []

    for subj, grp in wide_df.groupby("id"):
        t = grp["time"].values.astype(float) / 12.0   # months → years

        real_row = {"subject_id": subj}
        pred_row = {"subject_id": subj}

        for y_col, s_col in zip(y_cols, score_cols):
            roi_id = int(y_col.split("_")[-1])
            y_vals = grp[y_col].values.astype(float)
            s_vals = grp[s_col].values.astype(float)
            y_mask = ~np.isnan(y_vals)
            s_mask = ~np.isnan(s_vals)

            y_slope = (float(np.polyfit(t[y_mask], y_vals[y_mask], 1)[0])
                       if y_mask.sum() >= 2 else np.nan)
            s_slope = (float(np.polyfit(t[s_mask], s_vals[s_mask], 1)[0])
                       if s_mask.sum() >= 2 else np.nan)

            real_row[y_col] = y_slope
            pred_row[y_col] = s_slope   # same col names as y_ for downstream consistency

            qual_rows.append({
                "subject_id":     subj,
                "roi":            roi_id,
                "real_slope":     y_slope,
                "pred_slope":     s_slope,
                "num_timepoints": int(y_mask.sum()),
                "time_range":     (float(t[y_mask].max() - t[y_mask].min())
                                   if y_mask.sum() >= 2 else np.nan),
                "model":          "BrainGenFlow",
            })

        real_rows.append(real_row)
        pred_rows.append(pred_row)

    real_roc_df   = pd.DataFrame(real_rows)
    pred_roc_df   = pd.DataFrame(pred_rows)
    slope_qual_df = pd.DataFrame(qual_rows)
    return baseline_df, real_roc_df, pred_roc_df, slope_qual_df


def perform_statistical_comparison_braingen(
        baseline_results: dict,
        braingen_results: dict,
        real_results: dict,
        n_subj: int,
        alpha: float = 0.05) -> dict:
    """
    DeLong's test for the three pairwise comparisons:
        BrainGenFlow predicted RoC  vs  Baseline volumes
        BrainGenFlow predicted RoC  vs  Real RoC (upper bound)
        Real RoC (upper bound)      vs  Baseline volumes

    Requires that all three evaluate_features() calls used the same subject
    order and random_state so the pooled CV prediction vectors are aligned.
    """
    clf = BRAINGEN_CLASSIFIER

    y_true_base = baseline_results[clf]['all_y_true']
    y_true_bg   = braingen_results[clf]['all_y_true']
    y_true_real = real_results[clf]['all_y_true']

    assert np.array_equal(y_true_base, y_true_bg), \
        "y_true mismatch: baseline vs BrainGenFlow — subject ordering changed?"
    assert np.array_equal(y_true_base, y_true_real), \
        "y_true mismatch: baseline vs Real RoC — subject ordering changed?"
    y_true = y_true_bg

    y_prob_base = baseline_results[clf]['all_y_prob']
    y_prob_bg   = braingen_results[clf]['all_y_prob']
    y_prob_real = real_results[clf]['all_y_prob']

    comparisons = [
        ("BrainGenFlow predicted RoC",  "Baseline volumes",        y_prob_bg,   y_prob_base),
        ("BrainGenFlow predicted RoC",  "Real RoC (upper bound)",  y_prob_bg,   y_prob_real),
        ("Real RoC (upper bound)",      "Baseline volumes",        y_prob_real, y_prob_base),
    ]

    print("\n" + "=" * 80)
    print("DELONG'S TEST — BrainGenFlow feature comparisons")
    print(f"Classifier : {clf}   |   n ≈ {n_subj} MCI subjects (pooled CV)")
    print("=" * 80)

    print("\n  AUC summary (mean [95 % CI], 5 folds):")
    for label, res_dict in [("Real RoC (upper bound)",     real_results),
                             ("BrainGenFlow predicted RoC", braingen_results),
                             ("Baseline volumes",           baseline_results)]:
        m, lo, hi = compute_95ci(res_dict[clf]['fold_aucs'])
        print(f"    {label:<36}: {m:.4f}  [95% CI: {lo:.4f}, {hi:.4f}]")

    delong_out = {}
    for name_a, name_b, prob_a, prob_b in comparisons:
        key = f"{name_a} vs {name_b}"
        print(f"\n{'─' * 60}")
        print(f"  {key}")
        res = delong_test(y_true, prob_a, prob_b)
        delong_out[key] = res
        delta = res['auc_a'] - res['auc_b']
        sig   = '✓ Yes' if res['significant'] else '✗ No'
        print(f"  AUC ({name_a:<36}): {res['auc_a']:.4f}")
        print(f"  AUC ({name_b:<36}): {res['auc_b']:.4f}")
        print(f"  ΔAUC = {delta:+.4f}   z = {res['z_stat']:.3f}   "
              f"p = {res['p_value']:.4e}   sig (α={alpha}): {sig}")

    print(f"\n{'─' * 60}")
    print("CONCLUSION")
    for (name_a, name_b, _, _) in comparisons:
        key = f"{name_a} vs {name_b}"
        res = delong_out[key]
        delta = res['auc_a'] - res['auc_b']
        if not res['significant']:
            verdict = "does not significantly differ from"
        elif delta > 0:
            verdict = "significantly outperforms"
        else:
            verdict = "is significantly outperformed by"
        print(f"  {name_a} {verdict} {name_b} "
              f"(ΔAUC={delta:+.4f}, "
              f"z={res['z_stat']:.3f}, p={res['p_value']:.4e})")

    return delong_out


# =============================================================================
# MAIN — BrainGenFlow pipeline (merged_predictions_wide.csv only)
# =============================================================================
if __name__ == "__main__":
    WIDE_CSV = "./trajectory_error_analysis/merged_predictions_wide.csv"
    COV_CSV  = "./longitudinal_covariates_allstudies.csv"
    PRED_DIR = "./predictions"
    OUT_DIR  = "./mciprogression"
    LOG_FILE = os.path.join(OUT_DIR, "braingen_mci_progression_results.txt")
    CACHE    = os.path.join(OUT_DIR, "braingen_mci_cache.pkl")
    os.makedirs(OUT_DIR, exist_ok=True)

    import sys

    class _Tee:
        def __init__(self, path):
            self._f   = open(path, "w")
            self._out = sys.__stdout__
        def write(self, s):
            self._out.write(s); self._f.write(s)
        def flush(self):
            self._out.flush(); self._f.flush()
        def close(self):
            self._f.close()

    tee = _Tee(LOG_FILE)
    sys.stdout = tee

    try:
        print("=" * 80)
        print("MCI STABLE VS MCI PROGRESSOR CLASSIFICATION EXPERIMENT")
        print("=" * 80)
        print("Experiment Design:")
        print("  - Target: Discriminate MCI Stable (MCI→MCI) vs MCI Progressor (MCI→AD)")
        print("  - Data source: merged_predictions_wide.csv (BrainGenFlow outputs)")
        print("  - Feature sets compared:")
        print("      A. Baseline volumes      — 145 y_ ROIs at time=0 (from wide CSV)")
        print("      B. Real RoC              — OLS slope of y_ across observed timepoints")
        print("      C. BrainGenFlow pred RoC — OLS slope of full 70-month generated trajectory")
        print(f"  - Classifier: {BRAINGEN_CLASSIFIER}  (5-fold nested CV + inner 3-fold GridSearchCV)")
        print("=" * 80)

        # ── 1. Load merged predictions (baseline + real RoC) ─────────────────
        print("\n[1] Loading merged_predictions_wide.csv ...")
        wide = pd.read_csv(WIDE_CSV)
        print(f"  {len(wide):,} rows | {wide['id'].nunique()} subjects "
              f"| time {wide['time'].min():.0f}–{wide['time'].max():.0f} months")

        # ── 2. Baseline volumes + real RoC from wide CSV ──────────────────────
        print("\n[2] Extracting baseline volumes and real RoC from wide CSV ...")
        baseline_df, real_roc_df, _, slope_qual_df = compute_features_from_wide(wide)
        print(f"  Baseline (time=0):  {len(baseline_df)} subjects")
        print(f"  Real RoC (OLS y_):  {len(real_roc_df)} subjects")

        # ── 3. BrainGenFlow predicted RoC — full 70-month trajectory ─────────
        # The wide CSV only has rows at observed timepoints (2–4 per subject).
        # Slopes computed from so few points are noisy and miss the model's full
        # generated trajectory.  load_braingen_predictions() uses all 70 months
        # of the dense _gen output, giving a stable, representative slope.
        print("\n[3] Computing BrainGenFlow predicted RoC from full 70-month trajectories ...")
        braingen_long = load_braingen_predictions(PRED_DIR)

        # Pivot long → wide: subjects × 145 ROI pred_slopes
        pred_roc_df = (
            braingen_long
            .pivot(index="subject_id", columns="roi", values="pred_slope")
            .reset_index()
        )
        pred_roc_df.columns = (
            ["subject_id"] + [f"roi_{c}" for c in pred_roc_df.columns[1:]]
        )
        print(f"  Pred RoC (full traj): {len(pred_roc_df)} subjects, "
              f"{pred_roc_df.shape[1] - 1} ROIs")

        # Slope quality: real (sparse) vs predicted (full trajectory)
        print("\n[3b] BrainGenFlow slope quality ...")
        analyze_slope_quality(braingen_long)

        # ── 4. Progression labels ─────────────────────────────────────────────
        print("\n[4] Building MCI progression labels from covariates ...")
        cov     = pd.read_csv(COV_CSV)
        prog_df = build_progression_labels(cov)
        print(f"  MCI subjects with labels : {len(prog_df)}")
        print(f"  Progressors   (MCI→AD)   : {prog_df['is_progressor'].sum()}")
        print(f"  Non-progressors (MCI→MCI): {(prog_df['is_progressor'] == 0).sum()}")

        # ── 5. Intersect subjects across all sets ─────────────────────────────
        print("\n[5] Intersecting subjects across all feature sets ...")
        common = (
            set(baseline_df["subject_id"])
            & set(real_roc_df["subject_id"])
            & set(pred_roc_df["subject_id"])
            & set(prog_df["PTID"])
        )
        print(f"  Common subjects: {len(common)}")

        def _fs(df, id_col):
            return (df[df[id_col].isin(common)]
                    .sort_values(id_col).reset_index(drop=True))

        baseline_df = _fs(baseline_df, "subject_id")
        real_roc_df = _fs(real_roc_df, "subject_id")
        pred_roc_df = _fs(pred_roc_df, "subject_id")
        prog_df     = _fs(prog_df,     "PTID")

        y_labels = prog_df["is_progressor"].values
        n_subj   = len(y_labels)

        assert np.array_equal(baseline_df["subject_id"].values, prog_df["PTID"].values), \
            "Label alignment failed: baseline vs prog_df"
        assert np.array_equal(real_roc_df["subject_id"].values, prog_df["PTID"].values), \
            "Label alignment failed: real_roc vs prog_df"
        assert np.array_equal(pred_roc_df["subject_id"].values, prog_df["PTID"].values), \
            "Label alignment failed: pred_roc vs prog_df"
        print("  Label alignment verified ✓")
        print(f"  Final cohort: {n_subj} subjects  "
              f"({y_labels.sum()} progressors, "
              f"{(y_labels == 0).sum()} non-progressors, "
              f"{y_labels.mean() * 100:.1f}% progression rate)")

        base_feat_cols = [c for c in baseline_df.columns if c != "subject_id"]
        real_feat_cols = [c for c in real_roc_df.columns if c != "subject_id"]
        pred_feat_cols = [c for c in pred_roc_df.columns if c != "subject_id"]

        X_baseline = baseline_df[base_feat_cols].values
        X_real_roc = real_roc_df[real_feat_cols].fillna(0).values
        X_pred_roc = pred_roc_df[pred_feat_cols].fillna(0).values

        # ── 6. Evaluate all three feature sets ────────────────────────────────
        print("\n[6] Evaluating feature sets (5-fold nested CV) ...")
        baseline_results = evaluate_features(
            X_baseline, y_labels, "Baseline volumes (y_ at t=0)")
        real_results     = evaluate_features(
            X_real_roc,  y_labels, "Real RoC (upper bound)")
        braingen_results = evaluate_features(
            X_pred_roc,  y_labels, "BrainGenFlow predicted RoC")

        # Cache for plot-only re-runs
        with open(CACHE, "wb") as fh:
            pickle.dump({
                "baseline_results": baseline_results,
                "real_results":     real_results,
                "braingen_results": braingen_results,
                "n_subj":           n_subj,
            }, fh)
        print(f"\n  Results cached → {CACHE}")

        # ── 7. ROC figure ─────────────────────────────────────────────────────
        print("\n[7] Generating ROC figure ...")
        plot_comprehensive_roc_curves(
            baseline_results      = baseline_results,
            rnn_pred_results      = None,
            dkgp_pred_results     = None,
            real_results          = real_results,
            save_prefix           = "braingen_mci_progression",
            braingen_pred_results = braingen_results,
        )

        # ── 8. DeLong pairwise comparison ─────────────────────────────────────
        print("\n[8] Statistical comparison (DeLong's test) ...")
        perform_statistical_comparison_braingen(
            baseline_results = baseline_results,
            braingen_results = braingen_results,
            real_results     = real_results,
            n_subj           = n_subj,
        )

        # ── 9. Comprehensive metrics table ────────────────────────────────────
        clf  = BRAINGEN_CLASSIFIER
        sets = [
            ("Real RoC (upper bound)",      real_results[clf]),
            ("BrainGenFlow predicted RoC",  braingen_results[clf]),
            ("Baseline volumes",            baseline_results[clf]),
        ]
        metrics = [
            ("roc_auc",   "AUC-ROC"),
            ("pr_auc",    "AUC-PR"),
            ("accuracy",  "Accuracy"),
            ("precision", "Precision"),
            ("recall",    "Recall"),
            ("f1",        "F1"),
            ("fdr",       "FDR"),
            ("fpr_stat",  "FPR"),
        ]
        W = 26
        print(f"\n{'=' * 80}")
        print(f"COMPREHENSIVE METRICS  (mean ± std, {clf}, 5 folds)")
        print(f"{'=' * 80}")
        print(f"\n  {'Metric':<12}", end="")
        for name, _ in sets:
            print(f"  {name:<{W}}", end="")
        print()
        print("  " + "─" * (12 + (W + 2) * len(sets)))
        for key, label in metrics:
            print(f"  {label:<12}", end="")
            for _, r in sets:
                val  = r.get(key, float("nan"))
                std_key = f"{key}_std"
                cell = (f"{val:.3f}±{r[std_key]:.3f}" if std_key in r
                        else f"{val:.3f}")
                print(f"  {cell:<{W}}", end="")
            print()

        # Delta: BrainGenFlow predicted RoC vs other two
        bg_r   = braingen_results[clf]
        real_r = real_results[clf]
        base_r = baseline_results[clf]
        print(f"\n  Delta  BrainGenFlow − Baseline:")
        for key, label in metrics:
            bv = bg_r.get(key, float("nan"))
            bsv = base_r.get(key, float("nan"))
            if not (np.isnan(bv) or np.isnan(bsv)):
                print(f"    {label:<14}: {bv - bsv:+.3f}  "
                      f"{'↑' if bv > bsv else '↓'}")
        print(f"\n  Delta  BrainGenFlow − Real RoC (upper bound):")
        for key, label in metrics:
            bv = bg_r.get(key, float("nan"))
            rv = real_r.get(key, float("nan"))
            if not (np.isnan(bv) or np.isnan(rv)):
                print(f"    {label:<14}: {bv - rv:+.3f}  "
                      f"{'↑' if bv > rv else '↓'}")

        print(f"\n{'=' * 80}")
        print("DONE — outputs in", OUT_DIR)
        print(f"{'=' * 80}")

    finally:
        sys.stdout = sys.__stdout__
        tee.close()
        print(f"\nResults logged → {LOG_FILE}")


# ── Plot-only mode: regenerate figures from cached results ───────────────────
# Usage: python mci_progression_prediction.py --plot-only
if __name__ == "__main__" and len(__import__('sys').argv) > 1 \
        and __import__('sys').argv[1] == '--plot-only':
    import sys, pickle
    CACHE = "./mciprogression/braingen_mci_cache.pkl"
    if not os.path.exists(CACHE):
        print(f"✗ Cache not found at {CACHE}. Run the full script first.")
        sys.exit(1)
    print(f"Loading cached results from {CACHE} ...")
    with open(CACHE, "rb") as fh:
        cache = pickle.load(fh)
    os.makedirs("./mciprogression", exist_ok=True)
    plot_comprehensive_roc_curves(
        baseline_results      = cache["baseline_results"],
        rnn_pred_results      = None,
        dkgp_pred_results     = None,
        real_results          = cache["real_results"],
        save_prefix           = "braingen_mci_progression",
        braingen_pred_results = cache["braingen_results"],
    )
    print("Done. Check ./mciprogression/ for updated figures.")
