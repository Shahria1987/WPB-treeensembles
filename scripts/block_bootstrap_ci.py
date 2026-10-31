"""
Moving-block bootstrap 95% CIs for MAE and RMSE on monthly test predictions.

Design
------
* Unit of resampling = calendar month. All stations observed in a month are kept
  together, so cross-station (spatial) correlation within a month is preserved.
* Contiguous blocks of L months are drawn with replacement (circular moving-block
  bootstrap, Kunsch 1989; Politis & Romano 1992) until the resampled series has
  the original length T (72 months). This preserves autocorrelation and
  seasonality of the errors up to lag L.
* Primary block length L = 12 (one seasonal cycle). L = 6 and L = 24 are run as a
  sensitivity check.
* For each replicate, MAE and RMSE are recomputed from the pooled resampled
  (station, month) residuals. 95% CI = 2.5th / 97.5th percentiles (percentile CI).
* Per seed CIs are reported, plus a seed-averaged CI: the SAME resampled month
  indices are applied to all 5 seeds in each replicate and the metric is
  averaged across seeds, so the CI reflects test-set sampling variability of the
  seed-averaged score.

Usage
-----
python scripts/block_bootstrap_ci.py        # reads and writes ../predictions
"""
import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = {"temp": "temp_residual", "hum": "hum_residual"}


def load(pred_dir):
    frames = []
    for f in glob.glob(os.path.join(pred_dir, "seed*_*_predictions.csv")):
        m = re.search(r"seed(\d+)_(spatial|temporal)_predictions\.csv$", os.path.basename(f))
        if m:
            frames.append(pd.read_csv(f, parse_dates=["date"]))
    if not frames:
        raise FileNotFoundError(f"No seed*_*_predictions.csv files in {pred_dir}")
    df = pd.concat(frames, ignore_index=True)
    # recompute residuals from raw values (files round residuals to 4 dp)
    df["temp_residual"] = df["temp_true_raw"] - df["temp_pred_raw"]
    df["hum_residual"] = df["hum_true_raw"] - df["hum_pred_raw"]
    return df


def month_sums(df_split, seeds, months, col):
    """Per-month sums of |e| and e^2 and counts, shape (n_seeds, T)."""
    abs_s = np.zeros((len(seeds), len(months)))
    sq_s = np.zeros_like(abs_s)
    n = np.zeros_like(abs_s)
    midx = {m: i for i, m in enumerate(months)}
    for si, s in enumerate(seeds):
        d = df_split[df_split.seed == s]
        g = d.groupby("date")[col]
        a = g.apply(lambda x: np.abs(x).sum())
        q = g.apply(lambda x: (x ** 2).sum())
        c = g.size()
        for m in a.index:
            j = midx[m]
            abs_s[si, j], sq_s[si, j], n[si, j] = a[m], q[m], c[m]
    return abs_s, sq_s, n


def block_indices(T, L, B, rng):
    """Circular moving-block bootstrap index matrix, shape (B, T)."""
    k = int(np.ceil(T / L))
    starts = rng.integers(0, T, size=(B, k))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]) % T
    return idx.reshape(B, k * L)[:, :T]


def bootstrap(abs_s, sq_s, n, idx):
    # sums over resampled months -> (B, n_seeds)
    A = abs_s[:, idx].sum(axis=2).T
    Q = sq_s[:, idx].sum(axis=2).T
    N = n[:, idx].sum(axis=2).T
    mae = A / N
    rmse = np.sqrt(Q / N)
    return mae, rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default=str(Path(__file__).resolve().parent.parent / "predictions"))
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--block_lengths", type=int, nargs="+", default=[12, 6, 24])
    ap.add_argument("--rng_seed", type=int, default=20261029)
    args = ap.parse_args()
    out_dir = args.out_dir or args.pred_dir

    df = load(args.pred_dir)
    rows = []
    for split in sorted(df.split.unique()):
        dsp = df[df.split == split]
        seeds = sorted(dsp.seed.unique())
        months = sorted(dsp.date.unique())
        T = len(months)
        n_st = dsp.station_code.nunique()
        for tname, col in TARGETS.items():
            abs_s, sq_s, n = month_sums(dsp, seeds, months, col)
            mae_pt = abs_s.sum(1) / n.sum(1)
            rmse_pt = np.sqrt(sq_s.sum(1) / n.sum(1))
            for L in args.block_lengths:
                # same rng seed per (split, target, L) -> identical resamples across seeds
                rng = np.random.default_rng(args.rng_seed + L)
                idx = block_indices(T, L, args.B, rng)
                mae_b, rmse_b = bootstrap(abs_s, sq_s, n, idx)
                labels = [str(s) for s in seeds] + ["mean_of_seeds"]
                mae_b = np.column_stack([mae_b, mae_b.mean(1)])
                rmse_b = np.column_stack([rmse_b, rmse_b.mean(1)])
                mae_p = np.append(mae_pt, mae_pt.mean())
                rmse_p = np.append(rmse_pt, rmse_pt.mean())
                for j, lab in enumerate(labels):
                    for metric, pt, bs in (("MAE", mae_p[j], mae_b[:, j]), ("RMSE", rmse_p[j], rmse_b[:, j])):
                        lo, hi = np.percentile(bs, [2.5, 97.5])
                        rows.append(dict(split=split, target=tname, seed=lab, metric=metric,
                                         block_len_months=L, n_months=T, n_stations=n_st,
                                         n_obs=int(n[0].sum()), B=args.B,
                                         estimate=pt, ci_low=lo, ci_high=hi,
                                         ci_width=hi - lo, boot_se=bs.std(ddof=1),
                                         boot_mean=bs.mean()))
    res = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "block_bootstrap_ci_mae_rmse.csv")
    res.round(4).to_csv(out, index=False)

    main_tbl = res[(res.block_len_months == 12)]
    pd.set_option("display.width", 200)
    print(main_tbl[["split", "target", "seed", "metric", "estimate", "ci_low", "ci_high", "boot_se"]]
          .round(4).to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
