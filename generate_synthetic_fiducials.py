#!/usr/bin/env python3
"""
Step 2 - Generate synthetic sphere fiducials from the base reference data.

Input
  real_combined_detailed.csv from prepare_real_data.py
  (base reference: sphere_good_set_1.csv + sphere_bad_set_1.csv, 1 = Good, 0 = Bad)

Method (per class, in-volume rows only)
  1. K = Radius x PointZ is nearly constant for a sphere, so each class is modelled
     as a depth trend of K plus real residuals.
  2. Trend: median K per 100 mm bin, smoothed over 3 bins and interpolated.
     Beyond the deepest/shallowest real bin it is extended with the class's
     overall linear slope (flagged as Extrapolated).
  3. Residuals: drawn from real rows of the same class within +/-100 mm of the
     new depth (window widened until at least --min-pool rows), plus a small
     Gaussian jitter (--jitter x local residual std).
  4. K is clipped to the class's observed K range +/- --pad, so the gap between
     Good and Bad in the real data is kept.
  5. Radius = K / PointZ.

Samples
  - In volume (700-2800 mm): --n-per-class Good and --n-per-class Bad, PointZ uniform.
  - Out of range: --n-out-of-range below (--below-min..700) and the same number
    above (2800..--above-max), all labelled 0 (Bad). Their radius follows the
    Good or Bad model at random (50/50), so the model learns that depth alone
    makes them Bad.

Outputs (in --out-dir), notebook columns PointX, PointY, PointZ, Radius, GroundTruth
  synthetic_fiducials.csv           synthetic rows only
  synthetic_fiducials_detailed.csv  + Region, ModelClass, K, Extrapolated
  real_plus_synthetic.csv           real_combined.csv + synthetic_fiducials.csv
  synthetic_fidelity.png            real vs synthetic plots

Usage
  python generate_synthetic_fiducials.py --n-per-class 2500 --n-out-of-range 500 --seed 42
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

Z_MIN, Z_MAX = 700.0, 2800.0
# Folder of this script: data/ and model/ live next to it
HERE = os.path.dirname(os.path.abspath(__file__))
BIN = 100.0
NOTEBOOK_COLUMNS = ["PointX", "PointY", "PointZ", "Radius", "GroundTruth"]
CLASS_NAME = {0: "Bad", 1: "Good"}


class ClassModel:
    """Depth trend of K plus a pool of real residuals for one class."""

    def __init__(self, rows, pad, jitter, min_pool):
        self.z = rows.PointZ.to_numpy()
        self.k = rows.K.to_numpy()
        self.jitter = jitter
        self.min_pool = min_pool

        bins = np.floor(self.z / BIN) * BIN
        medians = pd.Series(self.k).groupby(bins).median()
        smoothed = medians.rolling(3, center=True, min_periods=1).median()
        self.bin_centres = smoothed.index.to_numpy() + BIN / 2
        self.bin_k = smoothed.to_numpy()
        self.slope = np.polyfit(self.z, self.k, 1)[0]
        self.z_lo, self.z_hi = self.bin_centres[0], self.bin_centres[-1]

        self.residuals = self.k - self.trend(self.z)
        self.k_lo = self.k.min() - pad
        self.k_hi = self.k.max() + pad

    def trend(self, z):
        z = np.asarray(z, dtype=float)
        k = np.interp(z, self.bin_centres, self.bin_k)
        k = np.where(z < self.z_lo, self.bin_k[0] + self.slope * (z - self.z_lo), k)
        k = np.where(z > self.z_hi, self.bin_k[-1] + self.slope * (z - self.z_hi), k)
        return k

    def residual_pool(self, z):
        window = BIN
        while True:
            pool = self.residuals[np.abs(self.z - z) <= window]
            if len(pool) >= self.min_pool or window > 5000:
                return pool
            window += BIN

    def sample(self, z_values, rng):
        k = self.trend(z_values)
        for i, z in enumerate(z_values):
            pool = self.residual_pool(z)
            k[i] += rng.choice(pool) + rng.normal(0.0, self.jitter * pool.std())
        return np.clip(k, self.k_lo, self.k_hi)

    def extrapolated(self, z_values):
        z_values = np.asarray(z_values)
        return (z_values < self.z.min()) | (z_values > self.z.max())


def generate(models, args, rng):
    parts = []

    for label, model in models.items():
        z = rng.uniform(Z_MIN, Z_MAX, args.n_per_class)
        k = model.sample(z, rng)
        parts.append(pd.DataFrame({
            "PointZ": z, "K": k, "GroundTruth": label, "Region": "in_volume",
            "ModelClass": CLASS_NAME[label], "Extrapolated": model.extrapolated(z),
        }))

    for region, lo, hi in [("below_volume", args.below_min, Z_MIN),
                           ("above_volume", Z_MAX, args.above_max)]:
        z = rng.uniform(lo, hi, args.n_out_of_range)
        model_class = rng.integers(0, 2, args.n_out_of_range)
        k = np.empty_like(z)
        for label, model in models.items():
            mask = model_class == label
            k[mask] = model.sample(z[mask], rng)
        parts.append(pd.DataFrame({
            "PointZ": z, "K": k, "GroundTruth": 0, "Region": region,
            "ModelClass": [CLASS_NAME[c] for c in model_class],
            "Extrapolated": True,
        }))

    synthetic = pd.concat(parts, ignore_index=True)
    synthetic["Radius"] = synthetic.K / synthetic.PointZ
    synthetic["PointX"] = np.nan
    synthetic["PointY"] = np.nan
    return synthetic.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)


def fidelity_report(real, synthetic, models):
    real_in = real[real.InVolume]
    synth_in = synthetic[synthetic.Region == "in_volume"]

    print("\nFidelity: K per 100 mm bin, real vs synthetic (in volume)")
    rows = []
    for label in (1, 0):
        r = real_in[real_in.GroundTruth == label]
        s = synth_in[synth_in.GroundTruth == label]
        for b in np.arange(Z_MIN, Z_MAX, BIN):
            rb = r[(r.PointZ >= b) & (r.PointZ < b + BIN)].K
            sb = s[(s.PointZ >= b) & (s.PointZ < b + BIN)].K
            rows.append({
                "Class": CLASS_NAME[label], "Bin": int(b),
                "real_n": len(rb), "synth_n": len(sb),
                "real_mean": rb.mean(), "synth_mean": sb.mean(),
                "mean_diff_%": 100 * (sb.mean() - rb.mean()) / rb.mean() if len(rb) else np.nan,
                "real_std": rb.std(), "synth_std": sb.std(),
            })
    table = pd.DataFrame(rows).round(1)
    print(table.to_string(index=False, na_rep="-"))

    print("\nKS test on K, real vs synthetic resampled to the real depth mix "
          "(statistic < ~0.1 = close match)")
    rng = np.random.default_rng(0)
    for label in (1, 0):
        r = real_in[real_in.GroundTruth == label]
        s = synth_in[(synth_in.GroundTruth == label)
                     & synth_in.PointZ.between(r.PointZ.min(), r.PointZ.max())]
        # Synthetic depth is uniform, real is not: weight each synthetic row by
        # how common its depth bin is in the real data before comparing.
        real_share = (np.floor(r.PointZ / BIN)).value_counts(normalize=True)
        synth_bin = np.floor(s.PointZ / BIN)
        weights = synth_bin.map(real_share).fillna(0) / synth_bin.map(synth_bin.value_counts())
        matched = rng.choice(s.K.to_numpy(), size=len(r), p=weights / weights.sum())
        stat, _ = ks_2samp(r.K, matched)
        print(f"  {CLASS_NAME[label]}: KS statistic {stat:.3f}")

    good_min = synth_in[synth_in.GroundTruth == 1].K.min()
    bad_max = synth_in[synth_in.GroundTruth == 0].K.max()
    print(f"\nClass margin in synthetic K: Bad max {bad_max:.0f} -> Good min "
          f"{good_min:.0f} (gap {good_min - bad_max:.0f}; real gap "
          f"{real_in[real_in.GroundTruth == 1].K.min() - real_in[real_in.GroundTruth == 0].K.max():.0f})")

    print("\nSynthetic rows by region and label")
    print(pd.crosstab(synthetic.Region, synthetic.GroundTruth)
          .rename(columns={0: "Bad (0)", 1: "Good (1)"}))
    extrap = synth_in[synth_in.Extrapolated]
    print(f"\nIn-volume synthetic rows beyond the real depth range (extrapolated): "
          f"{len(extrap)} (PointZ > {real_in.PointZ.max():.0f} or < {real_in.PointZ.min():.0f})")
    for label, model in models.items():
        print(f"  {CLASS_NAME[label]} trend slope {model.slope:+.3f} K/mm, "
              f"K clipped to {model.k_lo:.0f}-{model.k_hi:.0f}")


def plot(real, synthetic, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {1: "#2a78d6", 0: "#eb6834"}  # categorical slots 1 and 2
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor="#fcfcfb")

    def style(ax, title, ylabel):
        ax.set_facecolor("#fcfcfb")
        ax.axvspan(0, Z_MIN, color=grid, alpha=0.6, lw=0)
        ax.axvspan(Z_MAX, 4000, color=grid, alpha=0.6, lw=0)
        ax.set_xlim(min(real.PointZ.min(), synthetic.PointZ.min()) - 30,
                    synthetic.PointZ.max() + 30)
        ax.set_title(title, color=ink, loc="left", fontsize=12)
        ax.set_xlabel("PointZ (mm)  - shaded: outside 700-2800 mm, always Bad", color=muted)
        ax.set_ylabel(ylabel, color=muted)
        ax.grid(color=grid, lw=0.6)
        ax.tick_params(colors=muted)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)

    for ax, data, title in [(axes[0], real, f"Real ({len(real)} rows)"),
                            (axes[1], synthetic, f"Synthetic ({len(synthetic)} rows)")]:
        for label in (1, 0):
            d = data[data.GroundTruth == label]
            ax.scatter(d.PointZ, d.Radius, s=8, color=colours[label], alpha=0.55,
                       lw=0, label=f"{CLASS_NAME[label]} ({label})")
        style(ax, title, "Radius (mm)")
        ax.legend(frameon=False, labelcolor=ink)

    ax = axes[2]
    for label in (1, 0):
        for data, line, name in [(real[real.InVolume], "-", "real"),
                                 (synthetic[synthetic.Region == "in_volume"], "--", "synthetic")]:
            d = data[data.GroundTruth == label]
            med = d.groupby(np.floor(d.PointZ / BIN) * BIN + BIN / 2).K.median()
            ax.plot(med.index, med.values, line, color=colours[label], lw=2,
                    label=f"{CLASS_NAME[label]} {name}")
    style(ax, "Median K = Radius x PointZ per 100 mm bin", "K (mm²)")
    ax.legend(frameon=False, labelcolor=ink)

    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=os.path.dirname(HERE),
                        help="folder that holds CSV_data/ (default: parent of this script's folder)")
    parser.add_argument("--data-dir", default=None,
                        help="default: data/ next to this script")
    parser.add_argument("--out-dir", default=None, help="default: same as --data-dir")
    parser.add_argument("--n-per-class", type=int, default=2500,
                        help="in-volume synthetic rows per class")
    parser.add_argument("--n-out-of-range", type=int, default=500,
                        help="out-of-range rows on EACH side, labelled 0")
    parser.add_argument("--below-min", type=float, default=500.0)
    parser.add_argument("--above-max", type=float, default=3000.0)
    parser.add_argument("--pad", type=float, default=50.0,
                        help="how far K may go beyond each class's observed range")
    parser.add_argument("--jitter", type=float, default=0.1,
                        help="Gaussian jitter as a fraction of the local residual std")
    parser.add_argument("--min-pool", type=int, default=30,
                        help="minimum real residuals per depth window")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(HERE, "data")
    out_dir = args.out_dir or data_dir
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    real = pd.read_csv(os.path.join(data_dir, "real_combined_detailed.csv"))
    real_in = real[real.InVolume]
    print(f"Real rows: {len(real)} (in volume {len(real_in)}: "
          f"Good {int((real_in.GroundTruth == 1).sum())}, "
          f"Bad {int((real_in.GroundTruth == 0).sum())})")

    models = {label: ClassModel(real_in[real_in.GroundTruth == label],
                                args.pad, args.jitter, args.min_pool)
              for label in (1, 0)}
    synthetic = generate(models, args, rng)

    fidelity_report(real, synthetic, models)

    synthetic_path = os.path.join(out_dir, "synthetic_fiducials.csv")
    detailed_path = os.path.join(out_dir, "synthetic_fiducials_detailed.csv")
    combined_path = os.path.join(out_dir, "real_plus_synthetic.csv")
    plot_path = os.path.join(out_dir, "synthetic_fidelity.png")

    synthetic[NOTEBOOK_COLUMNS].to_csv(synthetic_path, index=False)
    synthetic[NOTEBOOK_COLUMNS + ["Region", "ModelClass", "K", "Extrapolated"]].to_csv(
        detailed_path, index=False)
    combined = pd.concat([real[NOTEBOOK_COLUMNS], synthetic[NOTEBOOK_COLUMNS]],
                         ignore_index=True)
    combined.to_csv(combined_path, index=False)
    plot(real, synthetic, plot_path)

    print()
    for path, table in [(synthetic_path, synthetic), (combined_path, combined)]:
        print(f"Saved {path} ({len(table)} rows, "
              f"Good {int((table.GroundTruth == 1).sum())}, "
              f"Bad {int((table.GroundTruth == 0).sum())})")
    print(f"Saved {detailed_path}")
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
