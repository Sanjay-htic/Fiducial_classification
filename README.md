# Fiducial_classification

Good/Bad classification of sphere fiducials seen by the FusionTrack500 tracker,
from two values per fiducial: **PointZ** (depth, mm) and **Radius** (mm).

- Labels: **1 = Good, 0 = Bad**
- Working volume: **700–2,800 mm**. Any fiducial outside it is Bad.
- Model: StandardScaler → RBF SVM, exported to ONNX for Python and C++ real-time use.

Only scripts are kept in this repository. The data CSVs and model files are produced
by the scripts (see [Run order](#run-order)) and are git-ignored.

## How it works

For a good sphere, `k = Radius × PointZ` stays almost constant with depth, and a bad
sphere has a lower k. In the base recordings:

| | k range |
|---|---|
| Good | 13,888 – 15,548 |
| Bad | 11,589 – 13,533 |

The boundary between them is a curve (Radius ≈ 13,700 / PointZ), so a linear model
cannot separate them (logistic regression ≈ 55%), while an RBF SVM can.

The base recordings are small and stop at 2,342 mm, so synthetic samples are added:
- **Inside the volume**: per class, a k-versus-depth trend is learnt from the real
  data and real residuals are resampled around it.
- **Outside the volume**: samples at 500–700 mm and 2,800–3,000 mm, all labelled 0.

## Repository layout

```
Fiducial_classification/
├── prepare_real_data.py            Step 1  base CSVs -> real / real_combined CSVs
├── generate_synthetic_fiducials.py Step 2  synthetic data -> real_plus_synthetic.csv
├── evaluate_augmentation.py        Step 3  check on held-out real data (optional)
├── compare_models.py               Step 3b logistic regression vs SVM (optional)
├── test_on_external.py             Step 3c test on other recorded CSVs (optional)
├── train_export_model.py           Step 4  train final SVM, export .pkl + .onnx
├── realtime_classifier.py          Step 5  Python real-time / replay classifier
├── cpp/
│   └── sphere_realtime_inference.cpp  Step 5  C++ real-time classifier (ONNX Runtime)
├── data/    (generated, not in git)
└── model/   (generated, not in git)
```

## Input data (not in git)

The scripts expect the repository to sit next to `CSV_data/`:

```
Fiducials_Good_Bad/
├── CSV_data/sphere/
│   ├── sphere_good_set_1.csv   base reference, all Good (valid = 0)
│   ├── sphere_bad_set_1.csv    base reference, all Bad  (valid = 1)
│   └── sphere_*_set_*.csv      other recordings, used only for testing
├── fiducial_level_training_dataset_001.csv   testing only
├── fiducial_level_training_dataset_002.csv   testing only
└── Fiducial_classification/    this repository
```

If `CSV_data/` is somewhere else, pass `--root <folder that holds CSV_data>`.

Only `sphere_good_set_1.csv` and `sphere_bad_set_1.csv` are used for training. In these
files `valid = 0` means Good, and they are converted to `GroundTruth = 1 − valid`.

## Requirements

Python 3.11 with:

```bash
pip install numpy pandas scipy scikit-learn joblib matplotlib skl2onnx onnxruntime requests
```

Versions used: Python 3.11, scikit-learn 1.6.1, skl2onnx 1.20.0, onnxruntime 1.23.2.
For C++: ONNX Runtime 1.20.1 (`onnxruntime-linux-x64-1.20.1`) and g++ with C++17.

## Run order

Run from the repository folder.

```bash
# Step 1: base CSVs -> data/real.csv, data/real_combined.csv, data/real_combined_detailed.csv
python prepare_real_data.py

# Step 2: synthetic data -> data/synthetic_fiducials.csv, data/real_plus_synthetic.csv,
#         data/synthetic_fidelity.png
python generate_synthetic_fiducials.py --n-per-class 2500 --n-out-of-range 500 --seed 42

# Step 3 (optional checks)
python evaluate_augmentation.py   # data/evaluation_per_fold.csv, evaluation_summary.csv
python compare_models.py          # data/model_comparison_*.csv
python test_on_external.py        # data/external_test_results.csv

# Step 4: train and export -> model/
python train_export_model.py
```

With the same seed, Steps 1–2 reproduce the same CSVs exactly, and Step 4 gives the same
model predictions.

### Generated files

| File | Rows | Good / Bad | Contents |
|---|---|---|---|
| `data/real.csv` | 3,213 | 2,258 / 955 | Real rows inside the volume |
| `data/real_combined.csv` | 3,240 | 2,258 / 982 | Real rows, out-of-range rows set to 0 |
| `data/synthetic_fiducials.csv` | 6,000 | 2,500 / 3,500 | Synthetic only |
| **`data/real_plus_synthetic.csv`** | **9,240** | **4,758 / 4,482** | **Training file** |

`*_detailed.csv` files carry extra columns (Source, Frame, InVolume, K).
The training CSVs have the notebook columns `PointX, PointY, PointZ, Radius, GroundTruth`,
so `model_training_sphere.ipynb` can also be trained on `real_plus_synthetic.csv`.

| File in `model/` | Use |
|---|---|
| `sphere_svm_scaler.onnx` + `sphere_svm_model.onnx` | Real-time use (Python or C++) |
| `sphere_svm_scaler.pkl` + `sphere_svm_model.pkl` | Python (joblib) |
| `model_metadata.json` | Labels, volume, parameters, ranges, verification |

## Using the ONNX model

Always use the scaler and model **as a pair from the same training run**.

| | Name | Type |
|---|---|---|
| Scaler input | `input` | float [N, 2], raw `[PointZ, Radius]` in mm |
| Scaler output → model input | `variable` → `input` | float [N, 2] |
| Model output | `output_label` | int64 [N], **1 = Good, 0 = Bad** |
| Model output | `output_probability` | float [N, 2], column 1 = P(Good) |

Before the model, mark any fiducial with PointZ outside 700–2,800 mm as Bad.

> The older `sphere_model.onnx` / `spherefiducial_model.onnx` / `spherefudicial_model.onnx`
> use **0 = Good** and are not compatible with this convention.

## Real-time classification

Each fiducial is checked every frame:
1. Depth check (700–2,800 mm).
2. Scaler → SVM gives P(Good).
3. Smoothing: median P(Good) over the last 5 frames. The state switches to Good at ≥ 0.7,
   to Bad at ≤ 0.3, and otherwise stays the same. A track resets after 10 missed frames.

### Python

```bash
# Live: polls the FusionTrack API
python realtime_classifier.py --live --url http://127.0.0.1:8081/GetCameraData

# Replay a recorded CSV (prints single-frame and smoothed accuracy)
python realtime_classifier.py --replay ../CSV_data/sphere/sphere_set_4.csv \
    --label-col valid --label-map true_is_good --fiducials-per-frame 4
```

Options: `--backend onnx|joblib`, `--window`, `--good-on`, `--bad-on`, `--interval`.

### C++

```bash
cd cpp
ORT=/path/to/onnxruntime-linux-x64-1.20.1
g++ -std=c++17 -O2 -I$ORT/include -L$ORT/lib -Wl,-rpath,$ORT/lib \
    -o sphere_realtime_inference sphere_realtime_inference.cpp -lonnxruntime

# The program loads the .onnx files from the current folder, so run it from model/
cd ../model
../cpp/sphere_realtime_inference --live 127.0.0.1 8081 /GetCameraData
../cpp/sphere_realtime_inference --manual            # type: 1000 14.8
../cpp/sphere_realtime_inference --batch in.csv out.csv
```

`--batch` reads `PointZ`/`point_z` and `Radius`/`radius` columns.
The C++ and Python outputs were checked on 27,850 readings: 0 label differences.

## Results

Confusion matrices are `[[TN, FP], [FN, TP]]`, with Good as positive.

| Test | SVM accuracy | Confusion matrix |
|---|---|---|
| Notebook 70/15/15 split, test part | 99.9% | [[673, 0], [1, 712]] |
| Held-out real data (5 folds by frame blocks) | 100% | [[982, 0], [0, 2,258]] |
| `sphere_val_set_1.csv` | 99.7% | [[1,251, 6], [0, 890]] |
| `sphere_processed_train_set_1.csv` | 100% | [[620, 0], [0, 310]] |
| `sphere_test_set_3.csv` (frame / smoothed) | 93.9% / 100% | [[230, 15], [0, 0]] |
| `sphere_set_4.csv` (frame / smoothed) | 76.1% / 88.1% | [[1,122, 352], [0, 0]] |
| `sphere_eval_set_1.csv` | 79.9% | [[785, 228], [0, 119]] |
| `sphere_test_set_2.csv` | 71.3% | [[77, 31], [0, 0]] |

- Best SVM: C = 1000, gamma = 1, 57 support vectors. Train and test scores are equal,
  so it is not overfitting.
- Logistic regression on the same data: about 55%.
- ONNX latency: about 0.015 ms per frame of 4 fiducials.

## Known limits

- The training data comes from one Good and one Bad recording. Other tools give 70–94%.
- Some bad spheres have a normal radius-to-depth ratio and are classified Good.
- Depths above 2,342 mm are covered only by extrapolated synthetic data.
- Labels in `fiducial_level_training_dataset_002.csv` look reversed and should be checked
  against the physical tool.
- Live mode has been tested only against a mock of the camera API, not the real camera.
