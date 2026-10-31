#!/usr/bin/env python3
"""
Step 3c - Train on real_plus_synthetic.csv, test on other recorded CSV files.

Training data
  data/real_plus_synthetic.csv
    = sphere_good_set_1.csv + sphere_bad_set_1.csv (real, volume rule applied)
    + synthetic rows generated from those two files only.

Models (as in model_training_sphere.ipynb, fit on the whole training file)
  LogisticRegression(random_state=42, max_iter=50000)
  SVC RBF via GridSearchCV (C [0.1..1000] x gamma [1..'auto'], scoring 'f1', cv=5)
  Features PointZ, Radius; StandardScaler fit on the training file.

Test files and how their labels map to GroundTruth 1 = Good
  0/1 'valid' or 'valid_ground' columns   0 = Good, 1 = Bad   (same as the base files)
  True/False 'valid' columns              True = Good, False = Bad
    (checked on sphere_val_set_1.csv, which has both: valid_ground 0 <-> valid_curvefit True)
  fiducial_level_training_dataset_001.csv GroundTruth 0 = Good (raw sessions of 27 Jul)
  fiducial_level_training_dataset_002.csv GroundTruth 1 = Good (raw Sphere/Good, Sphere/Bad)
  The volume rule is applied to every file: PointZ outside 700-2800 mm -> 0.

Skipped: sphere_good_set_1, sphere_bad_set_1, sphere_train_set (the training rows
themselves) and sphere_train_set_1 (raw API dump without labels).

Output: <data-dir>/external_test_results.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

Z_MIN, Z_MAX = 700.0, 2800.0
# Folder of this script: data/ and model/ live next to it
HERE = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["PointZ", "Radius"]
PARAM_GRID = {
    "C": [0.1, 1, 10, 100, 1000],
    "gamma": [1, 0.1, 0.01, 0.001, "scale", "auto"],
    "kernel": ["rbf"],
}
# Midpoint of the K gap in the base files (Bad max 13,533, Good min 13,888)
BASE_K_THRESHOLD = 13710.0

# file, z column, radius column, label column, mapping to 1 = Good
TEST_FILES = [
    ("CSV_data/sphere/sphere_val_set_1.csv", "point_z", "radius", "valid_ground", "zero_is_good"),
    ("CSV_data/sphere/sphere_processed_train_set_1.csv", "point_z", "radius", "valid", "zero_is_good"),
    ("CSV_data/sphere/sphere_eval_set_1.csv", "point_z", "radius", "valid", "true_is_good"),
    ("CSV_data/sphere/sphere_set_4.csv", "point_z", "radius", "valid", "true_is_good"),
    ("CSV_data/sphere/sphere_test_set_2.csv", "point_z", "radius", "valid", "true_is_good"),
    ("CSV_data/sphere/sphere_test_set_3.csv", "point_z", "radius", "valid", "true_is_good"),
    ("fiducial_level_training_dataset_001.csv", "PointZ", "Radius", "GroundTruth", "zero_is_good"),
    ("fiducial_level_training_dataset_002.csv", "PointZ", "Radius", "GroundTruth", "one_is_good"),
]


def load_test_file(root, path, z_col, r_col, label_col, mapping):
    raw = pd.read_csv(os.path.join(root, path))
    label = raw[label_col]
    if mapping == "true_is_good":
        good = label.astype(str).str.lower() == "true"
    elif mapping == "zero_is_good":
        good = label.astype(int) == 0
    else:
        good = label.astype(int) == 1
    df = pd.DataFrame({"PointZ": raw[z_col], "Radius": raw[r_col],
                       "OriginalGood": good.astype(int)})
    df["InVolume"] = df.PointZ.between(Z_MIN, Z_MAX)
    df["GroundTruth"] = np.where(df.InVolume, df.OriginalGood, 0)
    return df


def train_models(train, n_jobs):
    scaler = StandardScaler()
    X = scaler.fit_transform(train[FEATURES])
    y = train.GroundTruth
    logreg = LogisticRegression(random_state=42, max_iter=50000).fit(X, y)
    grid = GridSearchCV(SVC(), PARAM_GRID, scoring="f1", cv=5, n_jobs=n_jobs).fit(X, y)
    print(f"Logistic regression weights {np.round(logreg.coef_[0], 3)}, "
          f"bias {logreg.intercept_[0]:.3f}")
    print(f"SVM best {grid.best_params_}, CV F1 {grid.best_score_:.4f}, "
          f"support vectors {grid.best_estimator_.n_support_.sum()}")
    return scaler, {"logistic_regression": logreg, "svm_rbf": grid.best_estimator_}


def evaluate(model, scaler, test):
    X = scaler.transform(test[FEATURES])
    y = test.GroundTruth.to_numpy()
    pred = model.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    two_classes = len(np.unique(y)) == 2
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_acc": balanced_accuracy_score(y, pred) if two_classes else np.nan,
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0) if (y == 1).any() else np.nan,
        "f1": f1_score(y, pred, zero_division=0) if (y == 1).any() else np.nan,
        "roc_auc": roc_auc_score(y, model.decision_function(X)) if two_classes else np.nan,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=os.path.dirname(HERE),
                        help="folder that holds CSV_data/ (default: parent of this script's folder)")
    parser.add_argument("--data-dir", default=None,
                        help="default: data/ next to this script")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    data_dir = args.data_dir or os.path.join(HERE, "data")
    pd.set_option("display.width", 220)

    train = pd.read_csv(os.path.join(data_dir, "real_plus_synthetic.csv"))
    print(f"Training on real_plus_synthetic.csv: {len(train)} rows "
          f"(Good {int((train.GroundTruth == 1).sum())}, "
          f"Bad {int((train.GroundTruth == 0).sum())})")
    scaler, models = train_models(train, args.n_jobs)

    rows = []
    for path, z_col, r_col, label_col, mapping in TEST_FILES:
        test = load_test_file(args.root, path, z_col, r_col, label_col, mapping)
        inside = test[test.InVolume]
        # How well the file's labels agree with the base reference K boundary
        base_agree = ((inside.PointZ * inside.Radius >= BASE_K_THRESHOLD).astype(int)
                      == inside.GroundTruth).mean()
        info = {"file": os.path.basename(path), "rows": len(test),
                "good": int((test.GroundTruth == 1).sum()),
                "bad": int((test.GroundTruth == 0).sum()),
                "out_of_volume": int((~test.InVolume).sum()),
                "z_range": f"{test.PointZ.min():.0f}-{test.PointZ.max():.0f}",
                "labels_agree_with_base_K": base_agree}
        for name, model in models.items():
            rows.append({**info, "model": name, **evaluate(model, scaler, test)})

    results = pd.DataFrame(rows)
    out_path = os.path.join(data_dir, "external_test_results.csv")
    results.to_csv(out_path, index=False)

    print("\nTest files")
    print(results.drop_duplicates("file")[["file", "rows", "good", "bad", "out_of_volume",
                                           "z_range", "labels_agree_with_base_K"]]
          .round(3).to_string(index=False))
    for name in models:
        print(f"\n=== {name} ===")
        print(results[results.model == name][
            ["file", "accuracy", "balanced_acc", "precision", "recall", "f1", "roc_auc",
             "TN", "FP", "FN", "TP"]].round(4).to_string(index=False))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
