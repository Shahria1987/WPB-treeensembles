"""
run_lightgbm_default.py
"""

import sys, time, random, json, logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")


# ── Config ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent   # repository root
DATA_PATH  = ROOT / "data" / "Preprocessed_Weather_Data_final.csv"
# On Kaggle: DATA_PATH = "/kaggle/input/datasets/shahriahasan/weather-corpus/Preprocessed_Weather_Data_final.csv"
OUTPUT_DIR = ROOT / "results" / "default" / "lightgbm"

SEEDS   = [42, 7, 123, 2024, 999]
WINDOW  = 12

HOLDOUT_STATIONS = [10120, 10724, 11513, 11805, 12007]
TRAIN_YEAR_IDX = 51   # → 2012
VAL_YEAR_IDX   = 54   # → 2015

# EXACT raw_min and raw_max from Kabir
TEMP_MIN, TEMP_MAX = 16.95, 39.31
HUM_MIN,  HUM_MAX  = 39.81, 90.06

# LightGBM defaults (no tuning) — only set what's necessary for comparability.
# Everything else (num_leaves=31, learning_rate=0.1, max_depth=-1, min_child_samples=20,
# reg_alpha=0.0, reg_lambda=0.0, subsample=1.0, colsample_bytree=1.0, etc.) is left at
# LightGBM's factory defaults.
LGB_PARAMS = {
    "objective": "mae",
    "metric": "mae",
    "verbose": -1,
    "n_jobs": -1,
}
LGB_NUM_ITERATIONS  = 2000
LGB_EARLY_STOPPING  = 50


# ── Reproducibility ─────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)


def make_logger(name, log_path):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    for h in [logging.StreamHandler(sys.stdout),
              logging.FileHandler(log_path, mode="w")]:
        h.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(h)
    return logger


# ── Flat sequence construction ──────────────────────────────────────────────
def make_flat_sequences(df, feature_cols, target_cols, window):
    """Build (N, window*F) flat matrix. Window is months [i..i+11]; target is month i+12."""
    X_list, Y_list = [], []
    for _, grp in df.groupby("Station Code"):
        grp = grp.sort_values(["Year", "Month"]).reset_index(drop=True)
        if len(grp) <= window:
            continue
        f = grp[feature_cols].values.astype(np.float32)
        y = grp[target_cols].values.astype(np.float32)
        for i in range(len(grp) - window):
            X_list.append(f[i:i + window].flatten())
            Y_list.append(y[i + window])
    return np.array(X_list), np.array(Y_list)


# ── Rescaling ───────────────────────────────────────────────────────────────
def inverse_scale(norm, raw_min, raw_max):
    """Invert [0.2, 0.8] min-max back to raw units."""
    return raw_min + (norm - 0.2) / 0.6 * (raw_max - raw_min)


# ── Metrics (compute in RESCALED units to match paper) ─────────────────────
def compute_metrics(y_true_raw, y_pred_raw):
    return {
        "MAE":  round(float(mean_absolute_error(y_true_raw, y_pred_raw)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))), 4),
        "R2":   round(float(r2_score(y_true_raw, y_pred_raw)), 4),
        "MAPE": round(float(np.mean(np.abs((y_true_raw - y_pred_raw) /
                                           (np.abs(y_true_raw) + 1e-10))) * 100), 4),
    }


def evaluate_predictions(y_norm, p_norm, kind):
    """Returns both normalized-space and rescaled-space metrics."""
    if kind == "temp":
        y_raw = inverse_scale(y_norm, TEMP_MIN, TEMP_MAX)
        p_raw = inverse_scale(p_norm, TEMP_MIN, TEMP_MAX)
    else:
        y_raw = inverse_scale(y_norm, HUM_MIN, HUM_MAX)
        p_raw = inverse_scale(p_norm, HUM_MIN, HUM_MAX)
    return {
        "normalized": compute_metrics(y_norm, p_norm),
        "rescaled":   compute_metrics(y_raw, p_raw),
    }


# ── LightGBM trainer ────────────────────────────────────────────────────────
def train_lgb(X_tr, y_tr, X_vl, y_vl, seed, logger, target_name):
    params = dict(LGB_PARAMS)
    params["random_state"] = seed
    params["n_estimators"] = LGB_NUM_ITERATIONS

    logger.info(f"\n  Training LightGBM for {target_name} …")
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_vl, y_vl)],
        eval_metric="mae",
        callbacks=[
            lgb.early_stopping(LGB_EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    best_iter = int(model.best_iteration_) if model.best_iteration_ is not None else model.n_estimators
    # LightGBM stores best score under key 'l1' for MAE objective
    best_val  = float(model.best_score_["valid_0"]["l1"])
    logger.info(f"  → best iter: {best_iter}   best val MAE (norm): {best_val:.6f}")
    return model, best_iter, best_val


# ── Single seed run ─────────────────────────────────────────────────────────
def run_one(seed, df, feature_cols, target_cols,
            train_df, val_df, temp_test_df, spat_test_df, out_dir):
    set_seed(seed)
    tag = f"lightgbm_default_seed{seed}"
    logger = make_logger(tag, out_dir / f"{tag}.log")

    logger.info(f"{'='*72}")
    logger.info(f"  LightGBM (26 features)  —  seed={seed}  window={WINDOW}")
    logger.info(f"  Features: {len(feature_cols)} × window {WINDOW} = "
                f"{len(feature_cols)*WINDOW} flat")
    logger.info(f"  Rescale: temp [{TEMP_MIN}, {TEMP_MAX}], hum [{HUM_MIN}, {HUM_MAX}]")
    logger.info(f"{'='*72}")

    X_tr, y_tr = make_flat_sequences(train_df,     feature_cols, target_cols, WINDOW)
    X_vl, y_vl = make_flat_sequences(val_df,       feature_cols, target_cols, WINDOW)
    X_te, y_te = make_flat_sequences(temp_test_df, feature_cols, target_cols, WINDOW)
    X_sp, y_sp = make_flat_sequences(spat_test_df, feature_cols, target_cols, WINDOW)
    logger.info(f"Shapes: train={X_tr.shape}  val={X_vl.shape}  "
                f"t-test={X_te.shape}  s-test={X_sp.shape}")

    for name, arr in [("train", X_tr), ("val", X_vl), ("t-test", X_te), ("s-test", X_sp)]:
        if arr.size == 0:
            raise RuntimeError(f"{name} split EMPTY. Check split indices.")

    t0 = time.time()
    m_t, bi_t, bv_t = train_lgb(X_tr, y_tr[:, 0], X_vl, y_vl[:, 0], seed, logger, "Temperature")
    m_h, bi_h, bv_h = train_lgb(X_tr, y_tr[:, 1], X_vl, y_vl[:, 1], seed, logger, "Humidity")
    elapsed = time.time() - t0
    logger.info(f"\n  Training time: {elapsed/60:.2f} min")

    def eval_split(X, y, split_name):
        pt = m_t.predict(X); ph = m_h.predict(X)
        temp = evaluate_predictions(y[:, 0], pt, "temp")
        hum  = evaluate_predictions(y[:, 1], ph, "hum")
        logger.info(f"\n{'═'*72}\n  {split_name}\n{'═'*72}")
        logger.info(f"  {'Target':<13} {'Metric':<6} {'Normalized':>12} {'Rescaled':>12}")
        for tgt, m in [("Temperature", temp), ("Humidity", hum)]:
            for k in ["MAE", "RMSE", "R2", "MAPE"]:
                logger.info(f"  {tgt:<13} {k:<6} {m['normalized'][k]:>12.4f} "
                            f"{m['rescaled'][k]:>12.4f}")
        return temp, hum

    t_t, t_h = eval_split(X_te, y_te, f"TEMPORAL TEST — {tag}")
    s_t, s_h = eval_split(X_sp, y_sp, f"SPATIAL TEST  — {tag}")

    for h in logger.handlers[:]:
        h.close(); logger.removeHandler(h)

    return {
        "tag": tag, "model": "LightGBM-Default", "seed": seed, "window": WINDOW,
        "n_features": len(feature_cols), "n_flat_features": len(feature_cols)*WINDOW,
        "best_iter_temp": bi_t, "best_iter_hum": bi_h,
        "best_val_temp": bv_t, "best_val_hum": bv_h,
        "train_time_min": round(elapsed/60, 2),
        "rescale": {"temp": [TEMP_MIN, TEMP_MAX], "hum": [HUM_MIN, HUM_MAX]},
        "temporal": {"temperature": t_t, "humidity": t_h},
        "spatial":  {"temperature": s_t, "humidity": s_h},
    }


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    out_dir = Path(OUTPUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLightGBM version: {lgb.__version__}")
    print(f"Seeds: {SEEDS}   Window: {WINDOW}")

    print("Loading data …")
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values(["Station Code", "Year", "Month"]).reset_index(drop=True)

    TARGET_COLS = ["Temperature (Deg.Cel)", "Humidity (percent)"]
    EXCLUDE     = ["Date", "Station Code"]    # NOT features
    FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE]

    print(f"Excluded from features: {EXCLUDE}")
    print(f"Using {len(FEATURE_COLS)} features:")
    for i, c in enumerate(FEATURE_COLS, 1):
        mark = "  ← also target" if c in TARGET_COLS else ""
        print(f"  {i:2d}. {c}{mark}")
    print("  Note: raw Temp/Humidity are features because input window (months i..i+11)")
    print("  does not overlap with target month (i+12). Not data leakage.\n")

    years = sorted(df["Year"].unique())
    train_limit = years[TRAIN_YEAR_IDX]
    val_limit   = years[VAL_YEAR_IDX]
    print(f"Split thresholds (z-normalized Year): "
          f"train<={train_limit:.4f}  val<={val_limit:.4f}")

    train_df     = df[~df["Station Code"].isin(HOLDOUT_STATIONS) & (df["Year"] <= train_limit)]
    val_df       = df[~df["Station Code"].isin(HOLDOUT_STATIONS) &
                      (df["Year"] > train_limit) & (df["Year"] <= val_limit)]
    temp_test_df = df[~df["Station Code"].isin(HOLDOUT_STATIONS) & (df["Year"] > val_limit)]
    spat_test_df = df[ df["Station Code"].isin(HOLDOUT_STATIONS) & (df["Year"] > val_limit)]

    print(f"Splits: train={len(train_df):,}  val={len(val_df):,}  "
          f"t-test={len(temp_test_df):,}  s-test={len(spat_test_df):,}\n")

    all_results = []
    for seed in SEEDS:
        print(f"\n{'#'*72}\n  LightGBM (26f)  Seed={seed}\n{'#'*72}")
        r = run_one(seed, df, FEATURE_COLS, TARGET_COLS,
                    train_df, val_df, temp_test_df, spat_test_df, out_dir)
        all_results.append(r)
        (out_dir / "lightgbm_default_results.json").write_text(json.dumps(all_results, indent=2))

    print(f"\nDetailed per-seed results in: {out_dir}/lightgbm_default_results.json")
