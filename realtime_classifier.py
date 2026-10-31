#!/usr/bin/env python3
"""
Step 4b - Real-time sphere fiducial classifier.

Per fiducial, every frame
  1. Depth check: PointZ outside 700-2800 mm -> Bad at once (model not used).
  2. Model: exported scaler -> SVM (ONNX by default, or joblib) gives P(Good).
  3. Smoothing: median P(Good) over the last --window frames of that fiducial,
     with two thresholds: switch to Good at >= --good-on, to Bad at <= --bad-on,
     otherwise keep the previous state. Stops single noisy frames near the
     boundary from flipping the result.
  Labels: 1 = Good, 0 = Bad.

Modes
  Live (polls the FusionTrack API, like predict_fiducial.py):
    python realtime_classifier.py --live --url http://127.0.0.1:8081/GetCameraData
  Replay a recorded CSV as a frame stream (prints frame vs smoothed accuracy):
    python realtime_classifier.py --replay CSV_data/sphere/sphere_test_set_3.csv \
        --label-col valid --label-map true_is_good
"""
import argparse
import json
import os
import time
from collections import deque

import numpy as np
import pandas as pd

Z_MIN, Z_MAX = 700.0, 2800.0
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")


class SphereFiducialClassifier:
    def __init__(self, model_dir=DEFAULT_MODEL_DIR, backend="onnx", window=5,
                 good_on=0.7, bad_on=0.3, max_missing=10):
        self.backend = backend
        self.window, self.good_on, self.bad_on = window, good_on, bad_on
        self.max_missing = max_missing
        self.tracks = {}  # key -> {"probs": deque, "state": int, "last_frame": int}
        self.frame = 0
        if backend == "onnx":
            import onnxruntime as ort
            self.scaler = ort.InferenceSession(os.path.join(model_dir, "sphere_svm_scaler.onnx"))
            self.model = ort.InferenceSession(os.path.join(model_dir, "sphere_svm_model.onnx"))
        else:
            import joblib
            self.scaler = joblib.load(os.path.join(model_dir, "sphere_svm_scaler.pkl"))
            self.model = joblib.load(os.path.join(model_dir, "sphere_svm_model.pkl"))

    def _model_predict(self, X):
        """Returns (label, P(Good)) per row of raw [PointZ, Radius]."""
        if self.backend == "onnx":
            scaled = self.scaler.run(None, {"input": X.astype(np.float32)})[0]
            label, probability = self.model.run(None, {"input": scaled})
            return np.asarray(label), np.asarray(probability)[:, 1]
        scaled = self.scaler.transform(X)
        return self.model.predict(scaled), self.model.predict_proba(scaled)[:, 1]

    def classify_frame(self, fiducials):
        """fiducials: list of (key, point_z, radius). Returns one dict per fiducial."""
        self.frame += 1
        results = []
        if not fiducials:
            return results
        keys = [f[0] for f in fiducials]
        X = np.array([[f[1], f[2]] for f in fiducials], dtype=float)
        in_volume = (X[:, 0] >= Z_MIN) & (X[:, 0] <= Z_MAX)

        frame_label = np.zeros(len(X), dtype=int)
        p_good = np.zeros(len(X))
        if in_volume.any():
            frame_label[in_volume], p_good[in_volume] = self._model_predict(X[in_volume])

        for i, key in enumerate(keys):
            track = self.tracks.get(key)
            if track is None or self.frame - track["last_frame"] > self.max_missing:
                track = {"probs": deque(maxlen=self.window), "state": int(frame_label[i])}
                self.tracks[key] = track
            track["last_frame"] = self.frame
            track["probs"].append(p_good[i])
            smoothed = float(np.median(track["probs"]))
            if not in_volume[i]:
                track["state"] = 0
            elif smoothed >= self.good_on:
                track["state"] = 1
            elif smoothed <= self.bad_on:
                track["state"] = 0
            results.append({
                "key": key, "point_z": X[i, 0], "radius": X[i, 1],
                "in_volume": bool(in_volume[i]), "frame_label": int(frame_label[i]),
                "p_good": float(p_good[i]), "smoothed_p_good": smoothed,
                "state": track["state"],
            })
        return results


def fiducials_from_api(data):
    markers = data.get("RegisteredMarkersList", [])
    if isinstance(markers, str):
        markers = json.loads(markers)
    if isinstance(markers, dict):
        markers = [markers]
    fiducials = []
    for marker in markers:
        for index, fid in enumerate(marker.get("FiducialList", [])):
            fiducials.append((f"{marker.get('MarkerName')}:{index}",
                              fid["point"]["z"], fid["Radius"]))
    return fiducials


def run_live(classifier, url, interval):
    import requests
    print(f"Polling {url} (Ctrl+C to stop). 1 = Good, 0 = Bad")
    try:
        while True:
            response = requests.get(url, verify=False, timeout=5)
            if response.status_code != 200:
                print(f"Failed to fetch data. Status code: {response.status_code}")
            else:
                for r in classifier.classify_frame(fiducials_from_api(response.json())):
                    status = "Good" if r["state"] == 1 else "Bad"
                    reason = "" if r["in_volume"] else " (outside 700-2800 mm)"
                    print(f"Fiducial {r['key']}: z={r['point_z']:.1f} r={r['radius']:.3f} "
                          f"P(Good)={r['smoothed_p_good']:.2f} -> {status}{reason}")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped.")


def run_replay(classifier, path, label_col, label_map, per_frame):
    raw = pd.read_csv(path)
    z_col = "point_z" if "point_z" in raw else "PointZ"
    r_col = "radius" if "radius" in raw else "Radius"
    rows = []
    # Rows are taken in order; with --fiducials-per-frame k, row i belongs to fiducial i % k
    for start in range(0, len(raw), per_frame):
        chunk = raw.iloc[start:start + per_frame]
        fiducials = [(i % per_frame, z, r) for i, z, r in
                     zip(range(start, start + len(chunk)), chunk[z_col], chunk[r_col])]
        rows.extend(classifier.classify_frame(fiducials))
    out = pd.DataFrame(rows)
    if label_col:
        label = raw[label_col]
        if label_map == "true_is_good":
            good = label.astype(str).str.lower() == "true"
        elif label_map == "zero_is_good":
            good = label.astype(int) == 0
        else:
            good = label.astype(int) == 1
        truth = np.where(raw[z_col].between(Z_MIN, Z_MAX), good.astype(int), 0)
        print(f"{os.path.basename(path)}: {len(out)} fiducial readings | "
              f"frame accuracy {(out.frame_label == truth).mean():.4f} | "
              f"smoothed accuracy {(out.state == truth).mean():.4f}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--backend", choices=["onnx", "joblib"], default="onnx")
    parser.add_argument("--window", type=int, default=5, help="frames in the moving median")
    parser.add_argument("--good-on", type=float, default=0.7)
    parser.add_argument("--bad-on", type=float, default=0.3)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--url", default="http://127.0.0.1:8081/GetCameraData")
    parser.add_argument("--interval", type=float, default=0.1, help="seconds between polls")
    parser.add_argument("--replay", help="CSV file to replay")
    parser.add_argument("--label-col")
    parser.add_argument("--label-map", choices=["true_is_good", "zero_is_good", "one_is_good"],
                        default="zero_is_good")
    parser.add_argument("--fiducials-per-frame", type=int, default=1)
    args = parser.parse_args()

    classifier = SphereFiducialClassifier(args.model_dir, args.backend, args.window,
                                          args.good_on, args.bad_on)
    if args.live:
        run_live(classifier, args.url, args.interval)
    elif args.replay:
        run_replay(classifier, args.replay, args.label_col, args.label_map,
                   args.fiducials_per_frame)
    else:
        parser.error("use --live or --replay")


if __name__ == "__main__":
    main()
