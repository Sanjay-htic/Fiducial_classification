#!/usr/bin/env python3
"""
Step 3b - Logistic regression vs SVM on the prepared datasets, with overfitting checks.

Both models are exactly as in model_training_sphere.ipynb
  LogisticRegression(random_state=42, max_iter=50000)
  SVC RBF via GridSearchCV (same C / gamma grid, scoring 'f1', cv=5)
  Features PointZ, Radius; StandardScaler fit on train; GroundTruth 1 = Good.

Part A - notebook split
  The notebook's own 70/15/15 stratified random split (random_state=42) on
  real_combined.csv and real_plus_synthetic.csv. Reports train, validation and
  test scores so the train-test gap is visible.

Part B - held-out real data
  The contiguous frame-block folds from evaluate_augmentation.py (synthetic data
  regenerated from each fold's training rows), for both models, plus train
  accuracy, noisy real test and out-of-range probes.

Outputs (in --out-dir)
  model_comparison_notebook_split.csv, model_comparison_folds.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate_augmentation as ev  # noqa: E402

# Folder of this script: data/ and model/ live next to it
HERE = os.path.dirname(os.path.abspath(__file__))
FEATURES = ev.FEATURES


def fit(model_name, X_train, y_train, n_jobs):
    if model_name == "logistic_regression":
        model = LogisticRegression(random_state=42, max_iter=50000).fit(X_train, y_train)
        return model, {}
    grid = GridSearchCV(SVC(), ev.PARAM_GRID, scoring="f1", cv=5, n_jobs=n_jobs)
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    return model, {"C": grid.best_params_["C"], "gamma": grid.best_params_["gamma"],
                   "support_vectors": int(model.n_support_.sum())}


def metrics(model, X, y, prefix):
    pred = model.predict(X)
    out = {f"{prefix}_acc": accuracy_score(y, pred)}
    if len(np.unique(y)) == 2:
        out.update({
            f"{prefix}_precision": precision_score(y, pred, zero_division=0),
            f"{prefix}_recall": recall_score(y, pred, zero_division=0),
            f"{prefix}_f1": f1_score(y, pred, zero_division=0),
            f"{prefix}_auc": roc_auc_score(y, model.decision_function(X)),
        })
    return out


def part_a(data_dir, n_jobs):
    rows = []
    for name in ["real_combined.csv", "real_plus_synthetic.csv"]:
        df = pd.read_csv(os.path.join(data_dir, name))
        X, y = df[FEATURES], df.GroundTruth
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val, X_test = scaler.transform(X_val), scaler.transform(X_test)
        for model_name in ["logistic_regression", "svm_rbf"]:
            model, info = fit(model_name, X_train, y_train, n_jobs)
            rows.append({"dataset": name, "model": model_name, **info,
                         **metrics(model, X_train, y_train, "train"),
                         **metrics(model, X_val, y_val, "val"),
                         **metrics(model, X_test, y_test, "test")})
            if model_name == "logistic_regression":
                rows[-1]["weights"] = np.round(model.coef_[0], 3).tolist()
                rows[-1]["bias"] = round(float(model.intercept_[0]), 3)
    return pd.DataFrame(rows)


def part_b(data_dir, n_jobs, folds=5, blocks=20, seed=42):
    gen_args = ev.SimpleNamespace(n_per_class=2500, n_out_of_range=500,
                                  below_min=500.0, above_max=3000.0,
                                  pad=50.0, jitter=0.1, min_pool=30)
    real = ev.assign_folds(pd.read_csv(os.path.join(data_dir, "real_combined_detailed.csv")),
                           blocks, folds)
    rows = []
    for fold in range(folds):
        real_train, real_test = real[real.Fold != fold], real[real.Fold == fold]
        synthetic, models = ev.make_synthetic(real_train, gen_args, seed + fold)
        tests = {
            "real_test": real_test,
            "real_test_in_volume": real_test[real_test.InVolume],
            "real_noisy": ev.add_noise(real_test, np.random.default_rng(1000 + fold), 2.0, 5.0),
            "out_of_range": ev.out_of_range_probe(models, 500, 2000 + fold),
        }
        train_sets = {
            "real_only": real_train,
            "real_plus_synthetic": pd.concat([real_train, synthetic], ignore_index=True),
            "synthetic_only": synthetic,
        }
        for train_name, train in train_sets.items():
            scaler = StandardScaler()
            X_train = scaler.fit_transform(train[FEATURES])
            for model_name in ["logistic_regression", "svm_rbf"]:
                model, info = fit(model_name, X_train, train.GroundTruth, n_jobs)
                row = {"fold": fold, "train": train_name, "model": model_name, **info,
                       **metrics(model, X_train, train.GroundTruth, "train")}
                for test_name, test in tests.items():
                    row.update(metrics(model, scaler.transform(test[FEATURES]),
                                       test.GroundTruth, test_name))
                rows.append(row)
        print(f"fold {fold} done", flush=True)
    return pd.DataFrame(rows)


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
    pd.set_option("display.max_columns", 40)

    a = part_a(data_dir, args.n_jobs)
    a.to_csv(os.path.join(data_dir, "model_comparison_notebook_split.csv"), index=False)
    print("\n=== Part A: notebook 70/15/15 random split ===")
    cols = ["dataset", "model", "C", "gamma", "support_vectors", "train_acc", "val_acc",
            "test_acc", "test_precision", "test_recall", "test_f1", "test_auc"]
    print(a[cols].round(4).to_string(index=False))
    print(a.loc[a.model == "logistic_regression", ["dataset", "weights", "bias"]].to_string(index=False))

    b = part_b(data_dir, args.n_jobs)
    b.to_csv(os.path.join(data_dir, "model_comparison_folds.csv"), index=False)
    print("\n=== Part B: held-out real data, 5 frame-block folds (mean / worst fold) ===")
    cols = ["train_acc", "real_test_acc", "real_test_in_volume_acc", "real_test_f1",
            "real_test_auc", "real_noisy_acc", "out_of_range_acc"]
    print(b.groupby(["model", "train"])[cols].agg(["mean", "min"]).round(4).to_string())
    print("\nSVM support vectors / train rows (mean)")
    svm = b[b.model == "svm_rbf"].copy()
    svm["train_n"] = svm.train.map({"real_only": 2590, "real_plus_synthetic": 8590,
                                    "synthetic_only": 6000})
    print(svm.groupby("train").support_vectors.mean().round(0).to_string())


if __name__ == "__main__":
    main()
