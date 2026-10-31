import sys, time, random, json, logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings("ignore")


# ── Config ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent   # repository root
DATA_PATH  = ROOT / "data" / "Preprocessed_Weather_Data_final.csv"
# On Kaggle: DATA_PATH = "/kaggle/input/datasets/shahriahasan/weather-forecast-bd/Preprocessed_Weather_Data_final.csv"
OUTPUT_DIR = ROOT / "results" / "h2_h3"

SEEDS    = [42, 7, 123, 2024, 999]
WINDOW   = 12          # input look-back (unchanged)
HORIZONS = [2, 3]      # h=1 already done with the original script

HOLDOUT_STATIONS = [10120, 10724, 11513, 11805, 12007]
TRAIN_YEAR_IDX   = 51
VAL_YEAR_IDX     = 54

TEMP_MIN, TEMP_MAX = 16.95, 39.31
HUM_MIN,  HUM_MAX  = 39.81, 90.06

# ── Optuna best hyperparameters (tuned at h=1, reused for h=2 and h=3) ─────
CAT_TUNED_PARAMS_TEMP = {
    "learning_rate":       0.012758037180210544,
    "depth":               8,
    "l2_leaf_reg":         3.847811635133765,
    "bagging_temperature": 0.4140684194606276,
    "random_strength":     0.0033453865372889253,
    "min_data_in_leaf":    80,
    "rsm":                 0.6333709751024439,
}
CAT_TUNED_PARAMS_HUM = {
    "learning_rate":       0.008648828378800204,
    "depth":               8,
    "l2_leaf_reg":         1.7724554369785153,
    "bagging_temperature": 0.7997150432496581,
    "random_strength":     1.2099810816115689e-05,
    "min_data_in_leaf":    45,
    "rsm":                 0.6884245108472264,
}
CAT_FIXED_PARAMS = {
    "loss_function":        "MAE",
    "eval_metric":          "MAE",
    "verbose":              0,
    "allow_writing_files":  False,
    "thread_count":         -1,
}
CAT_NUM_ITERATIONS = 2500
CAT_EARLY_STOPPING = 50

METRICS = ["MAE", "RMSE", "R2", "MAPE"]


# ── Utilities ────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def make_logger(name, log_path):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    for h in [logging.StreamHandler(sys.stdout),
              logging.FileHandler(log_path, mode="w")]:
        h.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(h)
    return logger


# ── Flat sequence construction (direct multi-step) ───────────────────────────
def make_flat_sequences(df, feature_cols, target_cols, window, horizon):
    """
    Input  : months [i, i+window-1]   (12 months, all 26 features)
    Target : month  i + window + horizon - 1
             h=1 -> next month, h=2 -> 2 months ahead, h=3 -> 3 months ahead.
    The real calendar date is parsed from 'Date' ('Year'/'Month' are z-scored).
    """
    df = df.copy()
    df["_date_parsed"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    if df["_date_parsed"].isna().any():
        mask = df["_date_parsed"].isna()
        df.loc[mask, "_date_parsed"] = pd.to_datetime(df.loc[mask, "Date"], errors="coerce")

    X_list, Y_list, meta_list = [], [], []
    for station, grp in df.groupby("Station Code"):
        grp = grp.sort_values("_date_parsed").reset_index(drop=True)
        if len(grp) < window + horizon:
            continue
        f = grp[feature_cols].values.astype(np.float32)
        y = grp[target_cols].values.astype(np.float32)
        for i in range(len(grp) - window - horizon + 1):
            t = i + window + horizon - 1
            X_list.append(f[i:i + window].flatten())
            Y_list.append(y[t])
            target_date = grp.loc[t, "_date_parsed"]
            origin_date = grp.loc[i + window - 1, "_date_parsed"]
            meta_list.append({
                "station_code": station,
                "year":         int(target_date.year),
                "month":        int(target_date.month),
                "date":         target_date.strftime("%Y-%m-%d"),
                "origin_date":  origin_date.strftime("%Y-%m-%d"),
                "horizon":      horizon,
            })
    return np.array(X_list), np.array(Y_list), meta_list


# ── Rescaling & metrics ──────────────────────────────────────────────────────
def inverse_scale(norm, raw_min, raw_max):
    return raw_min + (norm - 0.2) / 0.6 * (raw_max - raw_min)


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "R2":   round(float(r2_score(y_true, y_pred)), 4),
        "MAPE": round(float(np.mean(np.abs((y_true - y_pred) /
                                           (np.abs(y_true) + 1e-10))) * 100), 4),
    }


def evaluate_predictions(y_norm, p_norm, kind):
    raw_min, raw_max = (TEMP_MIN, TEMP_MAX) if kind == "temp" else (HUM_MIN, HUM_MAX)
    y_raw = inverse_scale(y_norm, raw_min, raw_max)
    p_raw = inverse_scale(p_norm, raw_min, raw_max)
    return {
        "normalized": compute_metrics(y_norm, p_norm),
        "rescaled":   compute_metrics(y_raw, p_raw),
    }, y_raw, p_raw


# ── CatBoost trainer ─────────────────────────────────────────────────────────
def train_cat(X_tr, y_tr, X_vl, y_vl, seed, logger, target_name,
              tuned_params, model_save_path):
    params = dict(CAT_FIXED_PARAMS)
    params.update(tuned_params)
    params["random_seed"]           = seed
    params["iterations"]            = CAT_NUM_ITERATIONS
    params["early_stopping_rounds"] = CAT_EARLY_STOPPING

    logger.info(f"\n  Training CatBoost for {target_name} (seed={seed}) …")
    model = CatBoostRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=(X_vl, y_vl), verbose=200)
    best_iter = int(model.best_iteration_)
    best_val  = float(model.best_score_["validation"]["MAE"])

    logger.info(f"  → best iter: {best_iter}   best val MAE (norm): {best_val:.6f}")
    model.save_model(str(model_save_path))
    return model, best_iter, best_val


# ── Predictions CSV ───────────────────────────────────────────────────────────
def save_predictions_csv(meta, p_t_raw, p_h_raw, y_t_raw, y_h_raw,
                         split_name, seed, out_path):
    rows = []
    for i, m in enumerate(meta):
        rows.append({
            "station_code":  m["station_code"],
            "horizon":       m["horizon"],
            "origin_date":   m["origin_date"],
            "date":          m["date"],
            "year":          m["year"],
            "month":         m["month"],
            "temp_true_raw": round(float(y_t_raw[i]), 4),
            "temp_pred_raw": round(float(p_t_raw[i]), 4),
            "temp_residual": round(float(y_t_raw[i] - p_t_raw[i]), 4),
            "hum_true_raw":  round(float(y_h_raw[i]), 4),
            "hum_pred_raw":  round(float(p_h_raw[i]), 4),
            "hum_residual":  round(float(y_h_raw[i] - p_h_raw[i]), 4),
            "split":         split_name,
            "seed":          seed,
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)


# ── One (horizon, seed) run ───────────────────────────────────────────────────
def run_one(seed, horizon, feature_cols, target_cols,
            train_df, val_df, temp_test_df, spat_test_df, dirs):

    set_seed(seed)
    tag    = f"catboost_tuned_h{horizon}_seed{seed}"
    logger = make_logger(tag, dirs["logs"] / f"{tag}.log")

    logger.info(f"{'='*72}")
    logger.info(f"  CatBoost TUNED (26 features) — horizon={horizon}  seed={seed}  window={WINDOW}")
    logger.info(f"{'='*72}")

    X_tr, y_tr, _       = make_flat_sequences(train_df,     feature_cols, target_cols, WINDOW, horizon)
    X_vl, y_vl, _       = make_flat_sequences(val_df,       feature_cols, target_cols, WINDOW, horizon)
    X_te, y_te, meta_te = make_flat_sequences(temp_test_df, feature_cols, target_cols, WINDOW, horizon)
    X_sp, y_sp, meta_sp = make_flat_sequences(spat_test_df, feature_cols, target_cols, WINDOW, horizon)

    logger.info(f"Shapes: train={X_tr.shape}  val={X_vl.shape}  "
                f"t-test={X_te.shape}  s-test={X_sp.shape}")
    for name, arr in [("train", X_tr), ("val", X_vl), ("t-test", X_te), ("s-test", X_sp)]:
        if arr.size == 0:
            raise RuntimeError(f"{name} split EMPTY for horizon={horizon}.")

    temp_ckpt = dirs["models"] / f"h{horizon}_seed{seed}_temperature.cbm"
    hum_ckpt  = dirs["models"] / f"h{horizon}_seed{seed}_humidity.cbm"

    t0 = time.time()
    m_t, bi_t, bv_t = train_cat(X_tr, y_tr[:, 0], X_vl, y_vl[:, 0], seed, logger,
                                "Temperature", CAT_TUNED_PARAMS_TEMP, temp_ckpt)
    m_h, bi_h, bv_h = train_cat(X_tr, y_tr[:, 1], X_vl, y_vl[:, 1], seed, logger,
                                "Humidity",    CAT_TUNED_PARAMS_HUM,  hum_ckpt)
    elapsed = time.time() - t0
    logger.info(f"\n  Training time: {elapsed/60:.2f} min")

    def eval_and_save(X, y, meta, split):
        temp_m, y_t_raw, p_t_raw = evaluate_predictions(y[:, 0], m_t.predict(X), "temp")
        hum_m,  y_h_raw, p_h_raw = evaluate_predictions(y[:, 1], m_h.predict(X), "hum")

        logger.info(f"\n{'═'*72}\n  {split.upper()} TEST — {tag}\n{'═'*72}")
        logger.info(f"  {'Target':<13} {'Metric':<6} {'Normalized':>12} {'Rescaled':>12}")
        for tgt, m in [("Temperature", temp_m), ("Humidity", hum_m)]:
            for k in METRICS:
                logger.info(f"  {tgt:<13} {k:<6} {m['normalized'][k]:>12.4f} {m['rescaled'][k]:>12.4f}")

        out = dirs["preds"] / f"h{horizon}_seed{seed}_{split}_predictions.csv"
        save_predictions_csv(meta, p_t_raw, p_h_raw, y_t_raw, y_h_raw, split, seed, out)
        logger.info(f"  → predictions saved to {out}")
        return temp_m, hum_m

    t_t, t_h = eval_and_save(X_te, y_te, meta_te, "temporal")
    s_t, s_h = eval_and_save(X_sp, y_sp, meta_sp, "spatial")

    for h in logger.handlers[:]:
        h.close(); logger.removeHandler(h)

    return {
        "tag": tag, "model": "CatBoost-Tuned",
        "horizon": horizon, "seed": seed, "window": WINDOW,
        "n_train": int(len(X_tr)), "n_val": int(len(X_vl)),
        "n_test_temporal": int(len(X_te)), "n_test_spatial": int(len(X_sp)),
        "best_iter_temp": bi_t, "best_iter_hum": bi_h,
        "best_val_MAE_temp_norm": bv_t, "best_val_MAE_hum_norm": bv_h,
        "train_time_min": round(elapsed / 60, 2),
        "temporal": {"temperature": t_t, "humidity": t_h},
        "spatial":  {"temperature": s_t, "humidity": s_h},
    }


def print_summary(summary, title):
    print(f"\n{'='*72}\n  {title}\n{'='*72}")
    for (split, tgt), g in summary.groupby(level=[0, 1]):
        print(f"\n  {split.upper()} — {tgt}")
        print(f"  {'h':>3}  {'MAE':>16}  {'RMSE':>16}  {'R2':>16}  {'MAPE%':>16}")
        for idx, row in g.iterrows():
            h = idx[2]
            cells = [f"{row[(k,'mean')]:.4f}±{row[(k,'std')]:.4f}" for k in METRICS]
            print(f"  {h:>3}  " + "  ".join(f"{c:>16}" for c in cells))


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    out_dir = Path(OUTPUT_DIR)
    dirs = {
        "root":   out_dir,
        "preds":  ROOT / "predictions" / "h2_h3",
        "models": ROOT / "models" / "h2_h3",
        "logs":   out_dir / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    import catboost
    print(f"\nCatBoost version: {catboost.__version__}")
    print(f"Seeds: {SEEDS}   Window: {WINDOW}   Horizons: {HORIZONS}")

    df = pd.read_csv(DATA_PATH)
    df = df.sort_values(["Station Code", "Year", "Month"]).reset_index(drop=True)

    TARGET_COLS  = ["Temperature (Deg.Cel)", "Humidity (percent)"]
    EXCLUDE      = ["Date", "Station Code"]
    FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE]
    print(f"Using {len(FEATURE_COLS)} features (window={WINDOW} → {len(FEATURE_COLS)*WINDOW} flat)")

    # Same splits as the h=1 experiment
    years       = sorted(df["Year"].unique())
    train_limit = years[TRAIN_YEAR_IDX]
    val_limit   = years[VAL_YEAR_IDX]

    train_df     = df[~df["Station Code"].isin(HOLDOUT_STATIONS) & (df["Year"] <= train_limit)]
    val_df       = df[~df["Station Code"].isin(HOLDOUT_STATIONS) &
                      (df["Year"] > train_limit) & (df["Year"] <= val_limit)]
    temp_test_df = df[~df["Station Code"].isin(HOLDOUT_STATIONS) & (df["Year"] > val_limit)]
    spat_test_df = df[ df["Station Code"].isin(HOLDOUT_STATIONS) & (df["Year"] > val_limit)]

    experiment_config = {
        "model": "CatBoost-Tuned",
        "data_path": str(DATA_PATH), "seeds": SEEDS, "window": WINDOW,
        "horizons": HORIZONS,
        "multistep_strategy": "direct (one model per horizon)",
        "feature_names": FEATURE_COLS, "target_names": TARGET_COLS,
        "holdout_stations": HOLDOUT_STATIONS,
        "train_years": [y for y in years if y <= train_limit],
        "val_years":   [y for y in years if train_limit < y <= val_limit],
        "test_years":  [y for y in years if y > val_limit],
        "hyperparams": {
            "fixed": CAT_FIXED_PARAMS,
            "tuned_temp": CAT_TUNED_PARAMS_TEMP, "tuned_hum": CAT_TUNED_PARAMS_HUM,
            "iterations": CAT_NUM_ITERATIONS, "early_stop": CAT_EARLY_STOPPING,
            "note": "Optuna (seed=42, 50 trials) tuned at h=1; reused for h=2 and h=3",
        },
        "catboost_version": catboost.__version__,
    }
    (dirs["root"] / "experiment_config.json").write_text(json.dumps(experiment_config, indent=2))

    # ── Run h=2 and h=3 × 5 seeds ────────────────────────────────────────────
    all_results = []
    for horizon in HORIZONS:
        for seed in SEEDS:
            print(f"\n{'#'*72}\n  CatBoost TUNED (26f)  h={horizon}  seed={seed}\n{'#'*72}")
            r = run_one(seed, horizon, FEATURE_COLS, TARGET_COLS,
                        train_df, val_df, temp_test_df, spat_test_df, dirs)
            all_results.append(r)
            # checkpoint after every run
            (dirs["root"] / "metrics_results.json").write_text(json.dumps(all_results, indent=2))

    # ── Summary: mean ± std over seeds for each horizon ─────────────────────
    rows = []
    for r in all_results:
        for split in ["temporal", "spatial"]:
            for tgt in ["temperature", "humidity"]:
                rows.append({"split": split, "target": tgt, "horizon": r["horizon"],
                             "seed": r["seed"], **r[split][tgt]["rescaled"]})
    long_df = pd.DataFrame(rows)
    long_df.to_csv(dirs["root"] / "all_runs_rescaled.csv", index=False)
    summary = (long_df.groupby(["split", "target", "horizon"])[METRICS]
                      .agg(["mean", "std"]).round(4))
    summary.to_csv(dirs["root"] / "horizon_summary.csv")
    print_summary(summary, "h=2 / h=3 SUMMARY (rescaled, mean ± std over seeds)")
