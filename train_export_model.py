#!/usr/bin/env python3
"""
Step 4a - Train the final sphere SVM on real_plus_synthetic.csv and export it.

Model (as in model_training_sphere.ipynb)
  Features PointZ, Radius -> StandardScaler -> SVC(probability=True) RBF,
  GridSearchCV over C [0.1..1000] x gamma [1..'auto'], scoring 'f1', cv=5,
  fit on the whole training file. GroundTruth 1 = Good, 0 = Bad.

Exports (in --model-dir), same layout as the existing spherefudicial_* files
  sphere_svm_model.pkl, sphere_svm_scaler.pkl     joblib
  sphere_svm_model.onnx                           input 'input' float [N, 2] (scaled),
                                                  outputs 'output_label' (int64),
                                                  'output_probability' (float [N, 2],
                                                  column 1 = P(Good))
  sphere_svm_scaler.onnx                          input 'input' float [N, 2] (raw
                                                  PointZ, Radius), output 'variable'
  model_metadata.json                             training data, ranges, parameters

  NOTE: output_label 1 = Good. The older spherefudicial_* models and
  predict_fiducial.py treat 0 as Good.

Verification
  ONNX (scaler -> model) labels must equal the joblib model's labels on the
  training file and on every external test file; also prints prediction latency.
"""
import argparse
import json
import os
import sys
import time
from datetime import date

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_on_external as ext  # noqa: E402

# Folder of this script: data/ and model/ live next to it
HERE = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["PointZ", "Radius"]


def onnx_predict(scaler_session, model_session, X):
    scaled = scaler_session.run(None, {"input": X.astype(np.float32)})[0]
    label, probability = model_session.run(None, {"input": scaled})
    return label, probability[:, 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=os.path.dirname(HERE),
                        help="folder that holds CSV_data/ (default: parent of this script's folder)")
    parser.add_argument("--data-dir", default=None,
                        help="default: data/ next to this script")
    parser.add_argument("--model-dir", default=None,
                        help="default: model/ next to this script")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    data_dir = args.data_dir or os.path.join(HERE, "data")
    model_dir = args.model_dir or os.path.join(HERE, "model")
    os.makedirs(model_dir, exist_ok=True)

    # Train
    train = pd.read_csv(os.path.join(data_dir, "real_plus_synthetic.csv"))
    scaler = StandardScaler()
    # Fit on a plain array: real-time callers pass arrays, not DataFrames
    X_train = scaler.fit_transform(train[FEATURES].to_numpy())
    grid = GridSearchCV(SVC(probability=True, random_state=42), ext.PARAM_GRID,
                        scoring="f1", cv=5, n_jobs=args.n_jobs)
    grid.fit(X_train, train.GroundTruth)
    model = grid.best_estimator_
    print(f"Trained on real_plus_synthetic.csv ({len(train)} rows): best "
          f"{grid.best_params_}, CV F1 {grid.best_score_:.4f}, "
          f"support vectors {model.n_support_.sum()}")

    # Export
    paths = {name: os.path.join(model_dir, name) for name in [
        "sphere_svm_model.pkl", "sphere_svm_scaler.pkl", "sphere_svm_model.onnx",
        "sphere_svm_scaler.onnx", "model_metadata.json"]}
    joblib.dump(model, paths["sphere_svm_model.pkl"])
    joblib.dump(scaler, paths["sphere_svm_scaler.pkl"])
    input_type = [("input", FloatTensorType([None, 2]))]
    for obj, name in [(model, "sphere_svm_model.onnx"), (scaler, "sphere_svm_scaler.onnx")]:
        # zipmap off: output_probability is a plain float [N, 2] tensor, easy to read in C++
        # and the output names match the existing spherefudicial_model.onnx
        if obj is model:
            options = {id(model): {"zipmap": False}}
            final_types = [("output_label", Int64TensorType([None])),
                           ("output_probability", FloatTensorType([None, 2]))]
        else:
            options, final_types = None, None
        onx = convert_sklearn(obj, initial_types=input_type, options=options,
                              final_types=final_types, target_opset={"": 9, "ai.onnx.ml": 1})
        with open(paths[name], "wb") as f:
            f.write(onx.SerializeToString())

    # Verify ONNX against joblib on the training file and every external test file
    scaler_session = ort.InferenceSession(paths["sphere_svm_scaler.onnx"])
    model_session = ort.InferenceSession(paths["sphere_svm_model.onnx"])
    check_sets = {"real_plus_synthetic.csv": train}
    for path, z_col, r_col, label_col, mapping in ext.TEST_FILES:
        check_sets[os.path.basename(path)] = ext.load_test_file(
            args.root, path, z_col, r_col, label_col, mapping)

    print("\nVerification: ONNX vs joblib, and accuracy of the exported model")
    rows = []
    for name, df in check_sets.items():
        X = df[FEATURES].to_numpy()
        sk_label = model.predict(scaler.transform(X))
        onnx_label, onnx_good_prob = onnx_predict(scaler_session, model_session, X)
        rows.append({"file": name, "rows": len(df),
                     "onnx_label_mismatches": int((sk_label != onnx_label).sum()),
                     "accuracy_onnx": (onnx_label == df.GroundTruth.to_numpy()).mean()})
    verification = pd.DataFrame(rows)
    print(verification.round(4).to_string(index=False))

    # Latency: one frame of 4 fiducials, as in real time
    frame = train[FEATURES].to_numpy()[:4]
    for label, fn in [("joblib", lambda: model.predict(scaler.transform(frame))),
                      ("onnx", lambda: onnx_predict(scaler_session, model_session, frame))]:
        fn()
        start = time.perf_counter()
        for _ in range(500):
            fn()
        print(f"Latency {label}: {(time.perf_counter() - start) / 500 * 1000:.3f} ms "
              f"per frame of 4 fiducials")

    # Metadata
    good, bad = train[train.GroundTruth == 1], train[train.GroundTruth == 0]
    metadata = {
        "created": date.today().isoformat(),
        "label_convention": {"1": "Good", "0": "Bad",
                             "note": "older spherefudicial_* models use 0 = Good"},
        "features": FEATURES,
        "units": "mm",
        "working_volume_mm": {"min": 700.0, "max": 2800.0,
                              "rule": "PointZ outside the volume is Bad"},
        "training_file": "data/real_plus_synthetic.csv",
        "base_reference_files": ["CSV_data/sphere/sphere_good_set_1.csv",
                                 "CSV_data/sphere/sphere_bad_set_1.csv"],
        "training_rows": {"total": len(train), "good": len(good), "bad": len(bad)},
        "training_ranges": {
            cls: {"PointZ": [round(d.PointZ.min(), 2), round(d.PointZ.max(), 2)],
                  "Radius": [round(d.Radius.min(), 2), round(d.Radius.max(), 2)]}
            for cls, d in [("good", good), ("bad", bad)]},
        "real_data_depth_limit_mm": 2342.2,
        "model": {"type": "SVC", "kernel": "rbf", "probability": True,
                  "best_params": {k: v for k, v in grid.best_params_.items()},
                  "cv_f1": round(grid.best_score_, 4),
                  "support_vectors": int(model.n_support_.sum())},
        "scaler": {"mean": scaler.mean_.round(6).tolist(),
                   "scale": scaler.scale_.round(6).tolist()},
        "onnx": {"scaler": {"input": "input float[N,2] raw PointZ, Radius",
                            "output": "variable"},
                 "model": {"input": "input float[N,2] scaled",
                           "outputs": ["output_label int64 (1 = Good)",
                                       "output_probability float[N,2], column 1 = P(Good)"]}},
        "verification": verification.round(4).to_dict(orient="records"),
    }
    with open(paths["model_metadata.json"], "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print()
    for path in paths.values():
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
