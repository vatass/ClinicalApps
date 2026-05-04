'''
Merge per-subject trajectory prediction files (RNN-AD and MLP) with real
observed brain volumes into OldHarmonizedMUSEROIs format:

    id, time, fold, y_H_MUSE_Volume_<roi>, ..., score_H_MUSE_Volume_<roi>, ...

Input prediction directory layout:
    baselines/
        rnn/fold_0/trajectory_ptid_*.csv
            fold_1/trajectory_ptid_*.csv  …
        mlp/fold_0/trajectory_ptid_*.csv
            fold_1/trajectory_ptid_*.csv  …
    Rows  : one per ROI (0-indexed in ROI_Index column, 145 total)
    Cols  : Month_0, Month_1, ..., Month_N  (integer month offsets from baseline)

Real-values reference CSV (REAL_VALUES_FILE):
    Must have columns: id (or PTID), time (months from baseline),
    y_H_MUSE_Volume_<roi_id> for each of the 145 ROIs.

Time alignment:
    Real visits have integer month offsets. For each visit at month t,
    the prediction is extracted from Month_<t> in the trajectory file.
    If a subject's trajectory does not cover month t, that visit is skipped.

Usage:
    python merge_predictions.py
    Edit the CONFIGURATION block below to point at your data.
'''

import os
import re
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION — edit these paths before running
# =============================================================================

# Directory with fold_<N>/ subdirectories for each model
RNNAD_PRED_DIR = Path('./baselines/rnn')
MLP_PRED_DIR   = Path('./baselines/mlp')

# CSV with real observed brain volumes at each visit.
# Required columns: id (or PTID), time (months from baseline),
#                   y_H_MUSE_Volume_<roi_id> for each ROI.
REAL_VALUES_FILE = Path('./manuscript1/HarmonizedROIVolumes.csv')

# Maps 0-based ROI list index → actual MUSE Volume ID.
# If absent, column names fall back to y_H_MUSE_Volume_0, y_H_MUSE_Volume_1, …
HMUSE_FILE = Path('./hmuse_list.npy')

# Output files
OUT_RNNAD = Path('./merged_rnnad.csv')
OUT_MLP   = Path('./merged_mlp.csv')

# =============================================================================


def build_roi_id_map(hmuse_path: Path, n_rois: int) -> dict:
    '''Return {list_index: muse_volume_id}. Falls back to identity map.'''
    if hmuse_path.exists():
        hmuse = np.load(str(hmuse_path), allow_pickle=True)
        print(f'  hmuse_list.npy loaded → {len(hmuse)} entries')
        return {i: int(hmuse[i]) for i in range(min(n_rois, len(hmuse)))}
    print(f'  hmuse_list.npy not found — ROI column index used as MUSE Volume ID')
    return {i: i for i in range(n_rois)}


def load_real_values(real_file: Path) -> pd.DataFrame:
    '''
    Load the real observed brain volumes CSV. Normalises the subject-ID
    column to "id" and the time column to "time" (months).
    '''
    df = pd.read_csv(str(real_file))

    # Normalise subject-ID column name
    for col in df.columns:
        if col.strip().upper() in ('PTID', 'ID', 'SUBJECT_ID', 'SUBJECTID'):
            df = df.rename(columns={col: 'id'})
            break
    if 'id' not in df.columns:
        raise ValueError(
            f'Could not find a subject-ID column in {real_file}. '
            'Expected one of: id, PTID, subject_id, subjectid.'
        )

    # Normalise time column name
    for col in df.columns:
        if col.strip().upper() in ('TIME', 'VISIT_TIME', 'MONTH', 'MONTHS'):
            df = df.rename(columns={col: 'time'})
            break
    if 'time' not in df.columns:
        raise ValueError(
            f'Could not find a time column in {real_file}. '
            'Expected one of: time, Time, visit_time, month, months.'
        )

    y_cols = [c for c in df.columns if c.startswith('y_H_MUSE_Volume_')]
    if not y_cols:
        raise ValueError(
            f'No y_H_MUSE_Volume_* columns found in {real_file}.'
        )

    print(f'  Real values: {len(df):,} rows, {df["id"].nunique()} subjects, '
          f'{len(y_cols)} ROI columns')
    return df


def load_trajectory(fpath: Path) -> dict | None:
    '''
    Read one trajectory_ptid_*.csv file.
    Returns {roi_index: {month: value}} or None on error.
    '''
    try:
        df = pd.read_csv(str(fpath))
    except Exception as exc:
        print(f'    WARNING: could not read {fpath.name}: {exc}')
        return None

    if 'ROI_Index' not in df.columns:
        print(f'    WARNING: no ROI_Index column in {fpath.name}, skipping')
        return None

    month_cols = [c for c in df.columns if re.match(r'^Month_\d+$', c)]
    if not month_cols:
        print(f'    WARNING: no Month_* columns in {fpath.name}, skipping')
        return None

    month_indices = [int(c.split('_')[1]) for c in month_cols]

    trajectories = {}
    for _, row in df.iterrows():
        roi_idx = int(row['ROI_Index'])
        trajectories[roi_idx] = {
            m: float(row[col])
            for m, col in zip(month_indices, month_cols)
        }
    return trajectories


def merge_model_predictions(
    pred_dir: Path,
    real_df: pd.DataFrame,
    roi_id_map: dict,
    model_label: str,
) -> pd.DataFrame:
    '''
    For every subject with both a trajectory file and real-value rows,
    build one output row per visit in OldHarmonizedMUSEROIs format.

    Walks fold subdirectories (fold_0/, fold_1/, …) inside pred_dir and
    uses the directory index to populate the fold column.

    Output columns:
        id, time, fold, y_H_MUSE_Volume_<roi_id>, ..., score_H_MUSE_Volume_<roi_id>, ...
    '''
    # Collect (fold_number, filepath) pairs from fold<N> subdirectories
    fold_dirs = sorted(
        [d for d in pred_dir.iterdir() if d.is_dir() and re.match(r'^fold_\d+$', d.name)],
        key=lambda d: int(d.name[5:])
    )
    if not fold_dirs:
        raise FileNotFoundError(
            f'No fold_<N> subdirectories found in {pred_dir}. '
            'Expected layout: <pred_dir>/fold_0/, fold_1/, …'
        )

    fold_files: list[tuple[int, Path]] = []
    for fold_dir in fold_dirs:
        fold_num = int(fold_dir.name[5:])
        for fpath in sorted(fold_dir.glob('trajectory_ptid_*.csv')):
            fold_files.append((fold_num, fpath))

    print(f'  [{model_label}] {len(fold_dirs)} folds, '
          f'{len(fold_files)} trajectory files total in {pred_dir}')

    # Build MUSE Volume ID column name lists in stable order
    roi_indices = sorted(roi_id_map.keys())
    y_cols      = [f'y_H_MUSE_Volume_{roi_id_map[i]}'     for i in roi_indices]
    score_cols  = [f'score_H_MUSE_Volume_{roi_id_map[i]}' for i in roi_indices]

    # Index real values by subject id for fast lookup
    real_by_subj = {subj: grp for subj, grp in real_df.groupby('id')}

    rows = []
    skipped_no_real  = 0
    skipped_no_month = 0

    for fold, fpath in fold_files:
        ptid = re.sub(r'^trajectory_ptid_(.*)\.csv$', r'\1', fpath.name)

        if ptid not in real_by_subj:
            skipped_no_real += 1
            continue

        trajectories = load_trajectory(fpath)
        if trajectories is None:
            continue

        subj_real = real_by_subj[ptid]

        for _, visit in subj_real.iterrows():
            visit_month = int(round(float(visit['time'])))

            # Check that all ROIs have a prediction at this month
            missing = [
                i for i in roi_indices
                if i not in trajectories
                or visit_month not in trajectories[i]
            ]
            if missing:
                skipped_no_month += 1
                continue

            row = {'id': ptid, 'time': visit['time'], 'fold': fold}

            # Real values
            for i, col in zip(roi_indices, y_cols):
                muse_col = f'y_H_MUSE_Volume_{roi_id_map[i]}'
                row[col] = visit.get(muse_col, np.nan)

            # Predicted values
            for i, col in zip(roi_indices, score_cols):
                row[col] = trajectories[i][visit_month]

            rows.append(row)

    out_df = pd.DataFrame(rows, columns=['id', 'time', 'fold'] + y_cols + score_cols)

    print(f'  [{model_label}] Merged {len(out_df):,} visit rows '
          f'for {out_df["id"].nunique()} subjects')
    if skipped_no_real:
        print(f'  [{model_label}] Skipped {skipped_no_real} subjects '
              f'with no real-value rows')
    if skipped_no_month:
        print(f'  [{model_label}] Skipped {skipped_no_month} visits '
              f'where trajectory did not cover the visit month')

    return out_df


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    print('=' * 70)
    print('MERGE PREDICTIONS → OldHarmonizedMUSEROIs FORMAT')
    print('=' * 70)

    # Validate paths
    for path, label in [
        (RNNAD_PRED_DIR,   'RNNAD_PRED_DIR'),
        (MLP_PRED_DIR,     'MLP_PRED_DIR'),
        (REAL_VALUES_FILE, 'REAL_VALUES_FILE'),
    ]:
        if not path.exists():
            sys.exit(f'ERROR: {label} not found: {path}')

    print('\n[1/4] Loading hmuse ROI map...')
    # Infer n_rois from first trajectory file found inside any fold subdir
    sample_files = list(RNNAD_PRED_DIR.glob('fold_*/trajectory_ptid_*.csv'))
    if not sample_files:
        sample_files = list(MLP_PRED_DIR.glob('fold_*/trajectory_ptid_*.csv'))
    sample_df = pd.read_csv(str(sample_files[0]))
    n_rois    = int(sample_df['ROI_Index'].max()) + 1
    print(f'  Detected {n_rois} ROIs from sample trajectory file')

    roi_id_map = build_roi_id_map(HMUSE_FILE, n_rois)

    print('\n[2/4] Loading real observed brain volumes...')
    real_df = load_real_values(REAL_VALUES_FILE)

    print('\n[3/4] Merging RNN-AD predictions...')
    rnnad_df = merge_model_predictions(
        RNNAD_PRED_DIR, real_df, roi_id_map, 'RNN-AD'
    )
    rnnad_df.to_csv(str(OUT_RNNAD), index=False)
    print(f'  Saved → {OUT_RNNAD}')

    print('\n[4/4] Merging MLP predictions...')
    mlp_df = merge_model_predictions(
        MLP_PRED_DIR, real_df, roi_id_map, 'MLP'
    )
    mlp_df.to_csv(str(OUT_MLP), index=False)
    print(f'  Saved → {OUT_MLP}')

    print('\n' + '=' * 70)
    print('DONE')
    print(f'  RNN-AD : {OUT_RNNAD}  ({len(rnnad_df):,} rows, '
          f'{rnnad_df["id"].nunique()} subjects)')
    print(f'  MLP    : {OUT_MLP}  ({len(mlp_df):,} rows, '
          f'{mlp_df["id"].nunique()} subjects)')
    print('=' * 70)
