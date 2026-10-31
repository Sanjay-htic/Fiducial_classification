#!/usr/bin/env python3
"""
Step 3 - Does synthetic data help the notebook's SVM? Tested on held-out REAL data.

Model (same as model_training_sphere.ipynb)
  Features PointZ, Radius -> StandardScaler (fit on train) -> SVC RBF,
  GridSearchCV over C [0.1, 1, 10, 100, 1000] x gamma [1, 0.1, 0.01, 0.001,
  'scale', 'auto'], scoring 'f1', cv=5. GroundTruth 1 = Good.
  probability=False here: it only adds Platt scaling for predict_proba and does
  not change the fitted SVM or its predictions; ROC AUC uses decision_function.

Evaluation
  - Real data (real_combined_detailed.csv) is cut into contiguous time blocks of
    consecutive frames (--blocks per source file). Fold f holds out every block
    with block % --folds == f, so test frames are never neighbours of train frames.
  - Inside each fold the generator is refit on the training rows only, so no
    test information leaks into the synthetic data.
  - Three training sets per fold: real only, real + synthetic, synthetic only.
  - Tests, all on data the model never saw:
      real_test     held-out real rows (incl. real out-of-range rows, label 0)
      real_noisy    real_test with tracker-like noise on Radius and PointZ
      out_of_range  probe samples at 500-700 and 2800-3000 mm (label 0),
                    radius from both class models, from a separate seed

Outputs (in --out-dir)
  evaluation_per_fold.csv, evaluation_summary.csv

Usage
  python evaluate_augmentation.py --folds 5 --blocks 20
"""
import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_synthetic_fiducials as gen  # noqa: E402

# Folder of this script: data/ and model/ live next to it
HERE = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["PointZ", "Radius"]
PARAM_GRID = {
    "C": [0.1, 1, 10, 100, 1000],
    "gamma": [1, 0.1, 0.01, 0.001, "scale", "auto"],
    "kernel": ["rbf"],
}


def assign_folds(real, n_blocks, n_folds):
    """Contiguous frame blocks per source file, blocks spread across folds."""
    real = real.copy()
    sizes = real.groupby("Source").Frame.transform("size")
    real["Block"] = (real.Frame * n_blocks // sizes).astype(int)
    real["Fold"] = real.Block % n_folds
    return real


def make_synthetic(real_train, gen_args, seed):
    inside = real_train[real_train.InVolume]
    models = {label: gen.ClassModel(inside[inside.GroundTruth == label],
                                    gen_args.pad, gen_args.jitter, gen_args.min_pool)
              for label in (1, 0)}
    args = SimpleNamespace(**{**vars(gen_args), "seed": seed})
    return gen.generate(models, args, np.random.default_rng(seed)), models


def out_of_range_probe(models, n, seed):
    """Fresh label-0 samples outside the volume, radius from both class models."""
    args = SimpleNamespace(n_per_class=0, n_out_of_range=n, below_min=500.0,
                           above_max=3000.0, seed=seed)
    probe = gen.generate(models, args, np.random.default_rng(seed))
    return probe[probe.Region != "in_volume"]


def add_noise(test, rng, radius_pct, z_mm):
    noisy = test.copy()
    noisy["Radius"] = noisy.Radius * (1 + rng.normal(0, radius_pct / 100, len(noisy)))
    noisy["PointZ"] = noisy.PointZ + rng.normal(0, z_mm, len(noisy))
    return noisy


def fit_notebook_svm(train, n_jobs):
    scaler = StandardScaler()
    X = scaler.fit_transform(train[FEATURES])
    grid = GridSearchCV(SVC(), PARAM_GRID, scoring="f1", cv=5, n_jobs=n_jobs)
    grid.fit(X, train.GroundTruth)
    return scaler, grid.best_estimator_, grid.best_params_


def score(scaler, model, test):
    X = scaler.transform(test[FEATURES])
    y = test.GroundTruth.to_numpy()
    pred = model.predict(X)
    result = {"n": len(y), "accuracy": accuracy_score(y, pred)}
    if len(np.unique(y)) == 2:
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        result.update({
            "balanced_acc": balanced_accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, model.decision_function(X)),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        })
    else:
        # one-class test (out-of-range probes): accuracy = share predicted Bad
        result["FP"] = int((pred == 1).sum())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=os.path.dirname(HERE),
                        help="folder that holds CSV_data/ (default: parent of this script's folder)")
    parser.add_argument("--data-dir", default=None,
                        help="default: data/ next to this script")
    parser.add_argument("--out-dir", default=None, help="default: same as --data-dir")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--blocks", type=int, default=20,
                        help="contiguous frame blocks per source file")
    parser.add_argument("--n-per-class", type=int, default=2500)
    parser.add_argument("--n-out-of-range", type=int, default=500)
    parser.add_argument("--probe-n", type=int, default=500,
                        help="out-of-range probe samples per side")
    parser.add_argument("--noise-radius-pct", type=float, default=2.0,
                        help="Radius noise std, percent of radius")
    parser.add_argument("--noise-z-mm", type=float, default=5.0, help="PointZ noise std, mm")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(HERE, "data")
    out_dir = args.out_dir or data_dir
    gen_args = SimpleNamespace(n_per_class=args.n_per_class,
                               n_out_of_range=args.n_out_of_range,
                               below_min=500.0, above_max=3000.0,
                               pad=50.0, jitter=0.1, min_pool=30)

    real = assign_folds(pd.read_csv(os.path.join(data_dir, "real_combined_detailed.csv")),
                        args.blocks, args.folds)
    print(f"Real rows {len(real)}; {args.folds} folds of contiguous frame blocks")

    rows = []
    for fold in range(args.folds):
        real_train, real_test = real[real.Fold != fold], real[real.Fold == fold]
        synthetic, models = make_synthetic(real_train, gen_args, args.seed + fold)
        noisy = add_noise(real_test, np.random.default_rng(1000 + fold),
                          args.noise_radius_pct, args.noise_z_mm)
        probe = out_of_range_probe(models, args.probe_n, 2000 + fold)
        tests = {"real_test": real_test, "real_noisy": noisy, "out_of_range": probe}

        train_sets = {
            "real_only": real_train,
            "real_plus_synthetic": pd.concat([real_train, synthetic], ignore_index=True),
            "synthetic_only": synthetic,
        }
        for train_name, train in train_sets.items():
            start = time.time()
            scaler, model, params = fit_notebook_svm(train, args.n_jobs)
            for test_name, test in tests.items():
                rows.append({"fold": fold, "train": train_name, "test": test_name,
                             "train_n": len(train), "C": params["C"],
                             "gamma": params["gamma"], **score(scaler, model, test)})
            got = rows[-3]
            print(f"fold {fold} {train_name:20s} train {len(train):5d} "
                  f"C={params['C']} gamma={params['gamma']} | real_test acc "
                  f"{got['accuracy']:.4f} f1 {got.get('f1', np.nan):.4f} | "
                  f"{time.time() - start:.0f}s", flush=True)

    per_fold = pd.DataFrame(rows)
    metrics = ["accuracy", "balanced_acc", "precision", "recall", "f1", "roc_auc"]
    summary = (per_fold.groupby(["test", "train"])[metrics + ["FP", "FN"]]
               .agg(["mean", "min"]).round(4))
    per_fold.to_csv(os.path.join(out_dir, "evaluation_per_fold.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "evaluation_summary.csv"))

    pd.set_option("display.width", 200)
    for test_name in ["real_test", "real_noisy", "out_of_range"]:
        part = per_fold[per_fold.test == test_name]
        cols = ["accuracy"] if test_name == "out_of_range" else metrics
        table = part.groupby("train")[cols].agg(["mean", "min"]).round(4)
        totals = part.groupby("train")[[c for c in ["TN", "FP", "FN", "TP"]
                                        if c in part]].sum()
        print(f"\n=== {test_name} (mean and worst fold over {args.folds} folds) ===")
        print(table.to_string())
        print("Summed confusion counts (Good = positive):")
        print(totals.to_string())

    print("\nBest grid parameters per fold")
    print(per_fold[per_fold.test == "real_test"]
          .pivot(index="fold", columns="train", values=["C", "gamma"]).to_string())
    print(f"\nSaved {os.path.join(out_dir, 'evaluation_per_fold.csv')}")
    print(f"Saved {os.path.join(out_dir, 'evaluation_summary.csv')}")


if __name__ == "__main__":
    main()
