#!/usr/bin/env python3
"""
Step 1 - Prepare the real (base reference) fiducial dataset for the sphere model.

Base reference
  CSV_data/sphere/sphere_good_set_1.csv  (valid 0 = Good)
  CSV_data/sphere/sphere_bad_set_1.csv   (valid 1 = Bad)
fiducial_level_training_dataset_001.csv is NOT used.

Rules
  - Label convention: GroundTruth 1 = Good, 0 = Bad.
  - Working volume: 700 <= PointZ <= 2800 mm. Anything outside is Bad (0),
    whatever its original label.

Outputs (in --out-dir), notebook columns PointX, PointY, PointZ, Radius, GroundTruth
  real.csv                    in-volume rows only, original labels
  real_combined.csv           real.csv + out-of-volume rows labelled 0
  real_combined_detailed.csv  real_combined.csv + Source, Frame,
                              OriginalGroundTruth, InVolume, K (Radius x PointZ)
  PointX / PointY are not recorded in the base files and are left empty;
  the notebook only uses PointZ and Radius.

Usage
  python prepare_real_data.py --root /home/htic-nav-sw/Fiducials_Good_Bad
"""
import argparse
import os

import numpy as np
import pandas as pd

Z_MIN, Z_MAX = 700.0, 2800.0
NOTEBOOK_COLUMNS = ["PointX", "PointY", "PointZ", "Radius", "GroundTruth"]


def load_sphere_csv(path, source):
    """point_z/valid/radius files: valid 0 = Good, 1 = Bad."""
    raw = pd.read_csv(path)
    return pd.DataFrame({
        "PointX": np.nan,
        "PointY": np.nan,
        "PointZ": raw.point_z,
        "Radius": raw.radius,
        "OriginalGroundTruth": 1 - raw.valid.astype(int),
        "Source": source,
        "Frame": np.arange(len(raw)),
    })


def summarise(df):
    print("Rows per source and volume (after volume rule)")
    print(pd.crosstab([df.Source, df.InVolume], df.GroundTruth, margins=True)
          .rename(columns={0: "Bad (0)", 1: "Good (1)"}))

    relabelled = df[(df.OriginalGroundTruth == 1) & ~df.InVolume]
    print(f"\nGood rows relabelled Bad because they are outside "
          f"{Z_MIN:.0f}-{Z_MAX:.0f} mm: {len(relabelled)} "
          f"(PointZ {relabelled.PointZ.min():.1f}-{relabelled.PointZ.max():.1f})")

    inside = df[df.InVolume]
    print("\nInside the volume, per class")
    print(inside.groupby("GroundTruth")[["PointZ", "Radius", "K"]]
          .agg(["min", "median", "max"]).round(1).T.to_string())

    good_k = inside[inside.GroundTruth == 1].K
    bad_k = inside[inside.GroundTruth == 0].K
    print(f"\nK margin between classes: Bad max {bad_k.max():.0f} -> "
          f"Good min {good_k.min():.0f} (gap {good_k.min() - bad_k.max():.0f})")

    print("\nPer 100 mm depth bin inside the volume: count and Radius range")
    binned = inside.assign(Bin=(inside.PointZ // 100 * 100).astype(int))
    print(binned.groupby(["Bin", "GroundTruth"]).Radius
          .agg(["count", "min", "max"]).round(2).unstack("GroundTruth").to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="/home/htic-nav-sw/Fiducials_Good_Bad")
    parser.add_argument("--out-dir", default=None,
                        help="default: <root>/synthetic_pipeline/data")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.root, "synthetic_pipeline", "data")
    os.makedirs(out_dir, exist_ok=True)

    sphere_dir = os.path.join(args.root, "CSV_data", "sphere")
    df = pd.concat([
        load_sphere_csv(os.path.join(sphere_dir, "sphere_good_set_1.csv"), "sphere_good_set_1"),
        load_sphere_csv(os.path.join(sphere_dir, "sphere_bad_set_1.csv"), "sphere_bad_set_1"),
    ], ignore_index=True)
    df["InVolume"] = df.PointZ.between(Z_MIN, Z_MAX)
    df["GroundTruth"] = np.where(df.InVolume, df.OriginalGroundTruth, 0).astype(int)
    df["K"] = df.Radius * df.PointZ

    summarise(df)

    outputs = {
        "real.csv": df.loc[df.InVolume, NOTEBOOK_COLUMNS],
        "real_combined.csv": df[NOTEBOOK_COLUMNS],
        "real_combined_detailed.csv": df,
    }
    print()
    for name, table in outputs.items():
        path = os.path.join(out_dir, name)
        table.to_csv(path, index=False)
        print(f"Saved {path} ({len(table)} rows, "
              f"Good {int((table.GroundTruth == 1).sum())}, "
              f"Bad {int((table.GroundTruth == 0).sum())})")


if __name__ == "__main__":
    main()
