import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split, KFold


def create_splits(pat_ids, output_dir, percent_test=0.15, n_folds=5):
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    trainvalid_ids, test_ids = train_test_split(
        pat_ids, test_size=percent_test, random_state=42)
    test_ids = sorted(test_ids)
    print("Test cases", len(test_ids))
    pd.DataFrame(test_ids).to_csv(
        output_dir / "test_ids.csv", index=False, header=None)

    # now for the remaining ones do cross-validation
    cv_dir = output_dir / f"{n_folds}_fold_cv"
    if not cv_dir.exists():
        cv_dir.mkdir(parents=True)

    kf = KFold(n_splits=n_folds, random_state=42, shuffle=True)
    trainvalid_ids = np.array(trainvalid_ids)

    for fold, (train_idx, val_idx) in enumerate(kf.split(trainvalid_ids)):
        train_ids = sorted(trainvalid_ids[train_idx].tolist())
        val_ids = sorted(trainvalid_ids[val_idx].tolist())

        print(fold, len(train_ids), len(val_ids))

        pd.DataFrame(train_ids).to_csv(
            cv_dir / f"train_ids_fold_{fold}.csv", index=False, header=None)
        pd.DataFrame(val_ids).to_csv(
            cv_dir / f"valid_ids_fold_{fold}.csv", index=False, header=None)

    # return list_of_trains, list_of_vals, test_ids


# TODO: adjust this
plan_ids = [
    'plan_id1',
    'plan_id2',
    'plan_id3',
    # ...
]

split_dir = Path("../../../data/cv_splits")

create_splits(plan_ids, output_dir=split_dir, percent_test=0.15, n_folds=5)
