'''
Consolidate RNN and MLP baseline predictions into OldHarmonizedMUSEROIs format:

    id, time, fold, y_H_MUSE_Volume_<roi_id>, ..., score_H_MUSE_Volume_<roi_id>, ...

Input layout (baselines/):
    baselines/
        rnn/fold_0/  test_predictions.npy  (N_visits × 145, float32)
                     test_targets.npy      (N_visits × 145, float32)
                     subject_ids.json      list[str] length N_visits
        mlp/fold_0/  test_predictions.npy
                     test_targets.npy
                     test_ptids.json       list[str] length N_visits
                     stats.json            {mean: [...], std: [...]}  length 145+

    For MLP: predictions and targets are z-scored using per-fold stats.json.
             De-normalization is applied: value = z * std[:145] + mean[:145].
    For RNN: no stats file — values are used as-is (same scale as OldHarmonizedMUSEROIs).

Time alignment:
    OldHarmonizedMUSEROIs.csv is the reference for time values.
    For each subject, its visits are sorted by time in the reference CSV.
    The i-th occurrence of a PTID in the subject-ID list maps to the i-th
    (time-sorted) visit of that subject in the reference.

ROI mapping:
    hmuse_list.npy: position i → MUSE Volume ID.
    Column i in the prediction arrays → y/score_H_MUSE_Volume_<hmuse[i]>.
'''

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

BASELINES_DIR    = Path('./baselines')
REF_CSV          = Path('./OldHarmonizedMUSEROIs.csv')   # time reference
HMUSE_FILE       = Path('./hmuse_list.npy')              # ROI index → MUSE ID

OUT_RNN = Path('./merged_rnn.csv')
OUT_MLP = Path('./merged_mlp.csv')

# =============================================================================


def load_roi_ids(hmuse_path: Path, n_rois: int) -> list[int]:
    '''Return ordered list of MUSE Volume IDs (length n_rois).'''
    if hmuse_path.exists():
        hmuse = np.load(str(hmuse_path), allow_pickle=True)
        ids = [int(hmuse[i]) for i in range(min(n_rois, len(hmuse)))]
        print(f'  hmuse_list.npy → {len(ids)} ROI IDs')
        return ids
    print(f'  hmuse_list.npy not found — using 0-based indices as ROI IDs')
    return list(range(n_rois))


def load_reference_times(ref_csv: Path) -> dict[str, list[float]]:
    '''
    Return {ptid: [time_0, time_1, ...]} sorted ascending from
    OldHarmonizedMUSEROIs.csv.  Used to recover the time for each
    (ptid, visit_index) pair.
    '''
    print(f'  Loading reference times from {ref_csv} ...')
    ref = pd.read_csv(str(ref_csv), usecols=['id', 'time'])
    times = {}
    for ptid, grp in ref.groupby('id'):
        times[str(ptid)] = sorted(grp['time'].tolist())
    print(f'  Reference: {len(times)} subjects')
    return times


def load_fold(fold_dir: Path, ptid_file: str) -> tuple:
    '''
    Load one fold's arrays and subject IDs.

    Returns (ptids, targets, predictions, mean_or_None, std_or_None).
    targets and predictions have shape (N_visits, N_rois).
    mean/std are 1-D arrays of length >= N_rois, or None if no stats.json.
    '''
    preds   = np.load(str(fold_dir / 'test_predictions.npy'))
    targets = np.load(str(fold_dir / 'test_targets.npy'))

    with open(fold_dir / ptid_file) as f:
        ptids = json.load(f)

    assert len(ptids) == len(preds) == len(targets), (
        f'{fold_dir}: length mismatch — ptids {len(ptids)}, '
        f'preds {preds.shape}, targets {targets.shape}'
    )

    mean = std = None
    stats_path = fold_dir / 'stats.json'
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        mean = np.array(stats['mean'], dtype=np.float64)
        std  = np.array(stats['std'],  dtype=np.float64)

    return ptids, targets, preds, mean, std


def build_merged_df(
    model_dir: Path,
    ptid_file: str,
    ref_times: dict[str, list[float]],
    roi_ids: list[int],
) -> pd.DataFrame:
    '''
    Walk all fold_* subdirectories, load arrays, and build the merged DataFrame.
    '''
    fold_dirs = sorted(
        [d for d in model_dir.iterdir()
         if d.is_dir() and d.name.startswith('fold_')],
        key=lambda d: int(d.name.split('_')[1])
    )
    if not fold_dirs:
        raise FileNotFoundError(f'No fold_* subdirs found in {model_dir}')

    n_rois   = len(roi_ids)
    y_cols   = [f'y_H_MUSE_Volume_{r}'     for r in roi_ids]
    sc_cols  = [f'score_H_MUSE_Volume_{r}' for r in roi_ids]
    out_cols = ['id', 'time', 'fold'] + y_cols + sc_cols

    all_rows = []
    skipped_no_ref  = 0
    skipped_no_time = 0

    for fold_dir in fold_dirs:
        fold_num = int(fold_dir.name.split('_')[1])
        print(f'  fold_{fold_num}: loading ...')

        ptids, targets, preds, mean, std = load_fold(fold_dir, ptid_file)

        # De-normalize if stats available (MLP)
        n_cols = targets.shape[1]
        if mean is not None and std is not None:
            m = mean[:n_cols].astype(np.float64)
            s = std[:n_cols].astype(np.float64)
            targets = targets.astype(np.float64) * s + m
            preds   = preds.astype(np.float64)   * s + m
        else:
            targets = targets.astype(np.float64)
            preds   = preds.astype(np.float64)

        # Track visit index per subject within this fold
        visit_counter: dict[str, int] = defaultdict(int)

        for row_idx, ptid in enumerate(ptids):
            ptid = str(ptid)

            if ptid not in ref_times:
                skipped_no_ref += 1
                continue

            visit_idx = visit_counter[ptid]
            ref_list  = ref_times[ptid]

            if visit_idx >= len(ref_list):
                skipped_no_time += 1
                visit_counter[ptid] += 1
                continue

            time = ref_list[visit_idx]
            visit_counter[ptid] += 1

            row = {'id': ptid, 'time': time, 'fold': fold_num}
            for col_i, (yc, sc) in enumerate(zip(y_cols, sc_cols)):
                if col_i < n_cols:
                    row[yc] = targets[row_idx, col_i]
                    row[sc] = preds[row_idx,   col_i]
                else:
                    row[yc] = np.nan
                    row[sc] = np.nan

            all_rows.append(row)

        n_kept = sum(1 for r in all_rows if r['fold'] == fold_num)
        print(f'         {n_kept:,} rows kept')

    df = pd.DataFrame(all_rows, columns=out_cols)

    if skipped_no_ref:
        print(f'  Skipped {skipped_no_ref} rows: PTID not in reference CSV')
    if skipped_no_time:
        print(f'  Skipped {skipped_no_time} rows: visit index exceeds reference visits')

    return df


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    print('=' * 70)
    print('MERGE BASELINE PREDICTIONS → OldHarmonizedMUSEROIs FORMAT')
    print('=' * 70)

    for path, label in [
        (BASELINES_DIR, 'BASELINES_DIR'),
        (REF_CSV,       'REF_CSV'),
        (HMUSE_FILE,    'HMUSE_FILE'),
    ]:
        if not path.exists():
            sys.exit(f'ERROR: {label} not found: {path}')

    # Load reference times from OldHarmonizedMUSEROIs.csv
    print('\n[1/4] Loading reference times ...')
    ref_times = load_reference_times(REF_CSV)

    # Infer n_rois from first available fold
    sample_npy = next(BASELINES_DIR.glob('*/fold_0/test_predictions.npy'))
    n_rois = np.load(str(sample_npy), mmap_mode='r').shape[1]
    print(f'\n[2/4] Detected {n_rois} ROI columns from sample array')

    roi_ids = load_roi_ids(HMUSE_FILE, n_rois)

    print('\n[3/4] Processing RNN ...')
    rnn_df = build_merged_df(
        BASELINES_DIR / 'rnn', 'subject_ids.json', ref_times, roi_ids
    )
    rnn_df.to_csv(str(OUT_RNN), index=False)
    print(f'  Saved → {OUT_RNN}  '
          f'({len(rnn_df):,} rows, {rnn_df["id"].nunique()} subjects, '
          f'{rnn_df["fold"].nunique()} folds)')

    print('\n[4/4] Processing MLP ...')
    mlp_df = build_merged_df(
        BASELINES_DIR / 'mlp', 'test_ptids.json', ref_times, roi_ids
    )
    mlp_df.to_csv(str(OUT_MLP), index=False)
    print(f'  Saved → {OUT_MLP}  '
          f'({len(mlp_df):,} rows, {mlp_df["id"].nunique()} subjects, '
          f'{mlp_df["fold"].nunique()} folds)')

    print('\n' + '=' * 70)
    print('DONE')
    print('=' * 70)
