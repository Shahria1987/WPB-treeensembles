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
OUTPUT_DIR = ROOT / "results" / "tuned"

SEEDS   = [42, 7, 123, 2024, 999]
WINDOW  = 12

HOLDOUT_STATIONS = [10120, 10724, 11513, 11805, 12007]
TRAIN_YEAR_IDX   = 51
VAL_YEAR_IDX     = 54

TEMP_MIN, TEMP_MAX = 16.95, 39.31
HUM_MIN,  HUM_MAX  = 39.81, 90.06

# ── Optuna-discovered best hyperparameters ────────────────────────────────
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


# ── Reproducibility ──────────────────────────────────────────────────────────
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


# ── Flat sequence construction ───────────────────────────────────────────────
def make_flat_sequences(df, feature_cols, target_cols, window):
    """
    Returns:
        X          : (N, window*F) float32
        Y          : (N, 2)       float32
        meta       : list of dicts with station_code, year, month, date per sample

    NOTE: The 'Year' and 'Month' columns in the dataset are z-score normalized
    (e.g. 0.1445, 1.0136, -1.7054), so they cannot be used as human-readable
    metadata — int() casting them produces garbage like -1, 0, 1. The real
    calendar year and month are parsed from the 'Date' column instead.
    """
    df = df.copy()
    # Parse the Date column (format M/D/YYYY, e.g. "7/1/1961"). Day-of-month is
    # ignored since the data is monthly; only year and month are used.
    df["_date_parsed"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    if df["_date_parsed"].isna().any():
        # Fall back to flexible parsing if any rows fail the strict format.
        mask = df["_date_parsed"].isna()
        df.loc[mask, "_date_parsed"] = pd.to_datetime(
            df.loc[mask, "Date"], errors="coerce")

    X_list, Y_list, meta_list = [], [], []
    for station, grp in df.groupby("Station Code"):
        # Sort chronologically by the parsed real date (robust ordering).
        grp = grp.sort_values("_date_parsed").reset_index(drop=True)
        if len(grp) <= window:
            continue
        f = grp[feature_cols].values.astype(np.float32)
        y = grp[target_cols].values.astype(np.float32)
        for i in range(len(grp) - window):
            X_list.append(f[i:i + window].flatten())
            Y_list.append(y[i + window])
            target_date = grp.loc[i + window, "_date_parsed"]
            meta_list.append({
                "station_code": station,
                "year":  int(target_date.year),
                "month": int(target_date.month),
                "date":  target_date.strftime("%Y-%m-%d"),
            })
    return np.array(X_list), np.array(Y_list), meta_list


# ── Rescaling ────────────────────────────────────────────────────────────────
def inverse_scale(norm, raw_min, raw_max):
    return raw_min + (norm - 0.2) / 0.6 * (raw_max - raw_min)


def compute_metrics(y_true, y_pred):
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
    logger.info(f"    lr={params['learning_rate']:.6f}  depth={params['depth']}  "
                f"l2={params['l2_leaf_reg']:.4f}  rsm={params['rsm']:.4f}")

    model = CatBoostRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=(X_vl, y_vl), verbose=200)
    best_iter = int(model.best_iteration_)
    best_val  = float(model.best_score_["validation"]["MAE"])

    logger.info(f"  → best iter: {best_iter}   best val MAE (norm): {best_val:.6f}")
    model.save_model(str(model_save_path))
    logger.info(f"  → saved model to {model_save_path}")
    return model, best_iter, best_val


# ── CatBoost built-in importance (seed 42 only) ──────────────────────────────
def save_catboost_importance(model, feature_cols, window, out_path, logger):
    """
    Save CatBoost's BUILT-IN feature importance (PredictionValuesChange),
    aggregated across the input window. Kept for reference only.

    NOTE: this is NOT SHAP. The SHAP-based importances used in Table 7 and
    Fig. 7 are written by save_shap_values() below. Do not relabel this file
    as "mean |SHAP|".
    """
    raw_imp = model.get_feature_importance()          # length = window * F
    F = len(feature_cols)
    # Reshape to (window, F) and sum across window steps → (F,)
    imp_matrix = raw_imp.reshape(window, F)
    imp_summed  = imp_matrix.sum(axis=0)
    imp_norm    = imp_summed / (imp_summed.sum() + 1e-12) * 100.0

    df_imp = pd.DataFrame({
        "feature_name": feature_cols,
        "importance":   np.round(imp_norm, 6),
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    df_imp.to_csv(out_path, index=False)
    logger.info(f"  → CatBoost built-in importance saved to {out_path}")
    return df_imp


def _aggregate_shap_pct(shap_vals, feature_cols, window):
    """
    Aggregate flat (N, window*F) SHAP values into a per-feature % contribution.

    mean over samples of |SHAP|  →  sum across the time-window steps
                                  →  normalise to 100 %.
    Identical aggregation to paper_figures.shap_contributions(), so the CSV,
    Table 7, and Fig. 7 all report the same metric. Returns a sorted DataFrame.
    """
    F        = len(feature_cols)
    mean_abs = np.abs(shap_vals).mean(axis=0)          # (window*F,)  step-major
    per_feat = mean_abs.reshape(window, F).sum(axis=0) # (F,)
    pct      = per_feat / (per_feat.sum() + 1e-12) * 100.0
    return (pd.DataFrame({"feature_name": feature_cols,
                          "importance":   np.round(pct, 6)})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True))


# ── SHAP values (seed 42 only) ────────────────────────────────────────────────
def save_shap_values(model, X, feature_cols, window, out_path, imp_csv_path,
                     logger, max_samples=2000):
    """
    Compute and save SHAP values, and write the SHAP-based per-feature
    importance CSV used in Table 7 and Fig. 7.

    Uses a random subsample to keep runtime reasonable.
    Saved as .npz with keys: shap_values, feature_matrix, feature_names_flat.
    The CSV at imp_csv_path holds mean |SHAP| % per feature (sums to 100).
    """
    try:
        import shap
    except ImportError:
        logger.info("  ⚠  shap not installed — skipping SHAP export")
        return

    idx = np.random.choice(len(X), min(max_samples, len(X)), replace=False)
    X_sub = X[idx]

    logger.info(f"  Computing SHAP on {len(X_sub)} samples …")
    t0 = time.time()
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_sub)
    logger.info(f"  → SHAP done in {time.time()-t0:.1f}s")

    # Build flat feature names: feat_name_step0, feat_name_step1, …
    feat_names_flat = [f"{c}_step{s}" for s in range(window) for c in feature_cols]

    np.savez_compressed(
        str(out_path),
        shap_values=shap_vals.astype(np.float32),
        feature_matrix=X_sub.astype(np.float32),
        feature_names_flat=np.array(feat_names_flat),
        feature_names=np.array(feature_cols),
        window=np.array([window]),
    )
    logger.info(f"  → SHAP saved to {out_path}  shape={shap_vals.shape}")

    # SHAP-based feature importance CSV (this is the Table 7 / Fig 7 source)
    df_imp = _aggregate_shap_pct(shap_vals, feature_cols, window)
    df_imp.to_csv(imp_csv_path, index=False)
    logger.info(f"  → SHAP feature importance saved to {imp_csv_path}")


# ── Predictions CSV ───────────────────────────────────────────────────────────
def save_predictions_csv(meta, y, p_t_raw, p_h_raw, y_t_raw, y_h_raw,
                         split_name, seed, out_path):
    rows = []
    for i, m in enumerate(meta):
        rows.append({
            "station_code":  m["station_code"],
            "date":          m.get("date", ""),
            "year":          m["year"],
            "month":         m["month"],
            "temp_true_raw": round(float(y_t_raw[i]),  4),
            "temp_pred_raw": round(float(p_t_raw[i]),  4),
            "temp_residual": round(float(y_t_raw[i] - p_t_raw[i]), 4),
            "hum_true_raw":  round(float(y_h_raw[i]),  4),
            "hum_pred_raw":  round(float(p_h_raw[i]),  4),
            "hum_residual":  round(float(y_h_raw[i] - p_h_raw[i]),  4),
            "split":         split_name,
            "seed":          seed,
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)


# ── Single seed run ───────────────────────────────────────────────────────────
def run_one(seed, df, feature_cols, target_cols,
            train_df, val_df, temp_test_df, spat_test_df,
            out_dir, dirs, is_importance_seed):

    set_seed(seed)
    tag    = f"catboost_tuned_seed{seed}"
    logger = make_logger(tag, dirs["logs"] / f"{tag}.log")

    logger.info(f"{'='*72}")
    logger.info(f"  CatBoost TUNED (26 features)  —  seed={seed}  window={WINDOW}")
    logger.info(f"  Features: {len(feature_cols)} × window {WINDOW} = "
                f"{len(feature_cols)*WINDOW} flat")
    logger.info(f"{'='*72}")

    # ── Build splits ──────────────────────────────────────────────────────────
    X_tr, y_tr, _      = make_flat_sequences(train_df,     feature_cols, target_cols, WINDOW)
    X_vl, y_vl, _      = make_flat_sequences(val_df,       feature_cols, target_cols, WINDOW)
    X_te, y_te, meta_te = make_flat_sequences(temp_test_df, feature_cols, target_cols, WINDOW)
    X_sp, y_sp, meta_sp = make_flat_sequences(spat_test_df, feature_cols, target_cols, WINDOW)

    logger.info(f"Shapes: train={X_tr.shape}  val={X_vl.shape}  "
                f"t-test={X_te.shape}  s-test={X_sp.shape}")

    for name, arr in [("train", X_tr), ("val", X_vl), ("t-test", X_te), ("s-test", X_sp)]:
        if arr.size == 0:
            raise RuntimeError(f"{name} split EMPTY. Check split indices.")

    # ── Train ─────────────────────────────────────────────────────────────────
    temp_ckpt = dirs["models"] / f"seed{seed}_temperature.cbm"
    hum_ckpt  = dirs["models"] / f"seed{seed}_humidity.cbm"

    t0 = time.time()
    m_t, bi_t, bv_t = train_cat(X_tr, y_tr[:, 0], X_vl, y_vl[:, 0], seed, logger,
                                 "Temperature", CAT_TUNED_PARAMS_TEMP, temp_ckpt)
    m_h, bi_h, bv_h = train_cat(X_tr, y_tr[:, 1], X_vl, y_vl[:, 1], seed, logger,
                                 "Humidity",    CAT_TUNED_PARAMS_HUM,  hum_ckpt)
    elapsed = time.time() - t0
    logger.info(f"\n  Training time: {elapsed/60:.2f} min")

    # ── Feature importance + SHAP (seed 42 only) ──────────────────────────────
    if is_importance_seed:
        logger.info("\n  Saving CatBoost built-in importances (reference) …")
        save_catboost_importance(m_t, feature_cols, WINDOW,
                                 dirs["feat_imp"] / "temp_catboost_importance.csv", logger)
        save_catboost_importance(m_h, feature_cols, WINDOW,
                                 dirs["feat_imp"] / "hum_catboost_importance.csv",  logger)

        logger.info("\n  Saving SHAP values + SHAP importance (temporal test set) …")
        save_shap_values(m_t, X_te, feature_cols, WINDOW,
                         dirs["shap"]     / "shap_values_temp_temporal.npz",
                         dirs["feat_imp"] / "temp_feature_importance_temporal.csv", logger)
        save_shap_values(m_h, X_te, feature_cols, WINDOW,
                         dirs["shap"]     / "shap_values_hum_temporal.npz",
                         dirs["feat_imp"] / "hum_feature_importance_temporal.csv",  logger)

        logger.info("\n  Saving SHAP values + SHAP importance (spatial test set) …")
        save_shap_values(m_t, X_sp, feature_cols, WINDOW,
                         dirs["shap"]     / "shap_values_temp_spatial.npz",
                         dirs["feat_imp"] / "temp_feature_importance_spatial.csv", logger)
        save_shap_values(m_h, X_sp, feature_cols, WINDOW,
                         dirs["shap"]     / "shap_values_hum_spatial.npz",
                         dirs["feat_imp"] / "hum_feature_importance_spatial.csv",  logger)

    # ── Evaluate + save predictions ───────────────────────────────────────────
    def eval_and_save(X, y, meta, split_label):
        pt_norm = m_t.predict(X)
        ph_norm = m_h.predict(X)

        temp_metrics, y_t_raw, p_t_raw = evaluate_predictions(y[:, 0], pt_norm, "temp")
        hum_metrics,  y_h_raw, p_h_raw = evaluate_predictions(y[:, 1], ph_norm, "hum")

        logger.info(f"\n{'═'*72}\n  {split_label}\n{'═'*72}")
        logger.info(f"  {'Target':<13} {'Metric':<6} {'Normalized':>12} {'Rescaled':>12}")
        for tgt, m in [("Temperature", temp_metrics), ("Humidity", hum_metrics)]:
            for k in ["MAE", "RMSE", "R2", "MAPE"]:
                logger.info(f"  {tgt:<13} {k:<6} {m['normalized'][k]:>12.4f} "
                            f"{m['rescaled'][k]:>12.4f}")

        csv_name = f"seed{seed}_temporal_predictions.csv" \
                   if "TEMPORAL" in split_label else f"seed{seed}_spatial_predictions.csv"
        save_predictions_csv(
            meta, y, p_t_raw, p_h_raw, y_t_raw, y_h_raw,
            "temporal" if "TEMPORAL" in split_label else "spatial",
            seed, dirs["preds"] / csv_name
        )
        logger.info(f"  → predictions saved to {dirs['preds'] / csv_name}")
        return temp_metrics, hum_metrics

    t_t, t_h = eval_and_save(X_te, y_te, meta_te, f"TEMPORAL TEST — {tag}")
    s_t, s_h = eval_and_save(X_sp, y_sp, meta_sp, f"SPATIAL TEST  — {tag}")

    for h in logger.handlers[:]:
        h.close(); logger.removeHandler(h)

    return {
        # ── Identity ──────────────────────────────────────────────────────────
        "tag":           tag,
        "model":         "CatBoost-Tuned",
        "seed":          seed,
        "window":        WINDOW,
        "n_features":    len(feature_cols),
        "n_flat_feat":   len(feature_cols) * WINDOW,
        # ── Training info ─────────────────────────────────────────────────────
        "best_iter_temp":  bi_t,
        "best_iter_hum":   bi_h,
        "best_val_MAE_temp_norm": bv_t,
        "best_val_MAE_hum_norm":  bv_h,
        "train_time_min":  round(elapsed / 60, 2),
        "hyperparams": {
            "temperature": {**CAT_FIXED_PARAMS, **CAT_TUNED_PARAMS_TEMP,
                            "iterations": CAT_NUM_ITERATIONS,
                            "early_stopping_rounds": CAT_EARLY_STOPPING,
                            "random_seed": seed},
            "humidity":    {**CAT_FIXED_PARAMS, **CAT_TUNED_PARAMS_HUM,
                            "iterations": CAT_NUM_ITERATIONS,
                            "early_stopping_rounds": CAT_EARLY_STOPPING,
                            "random_seed": seed},
        },
        # ── Metrics ───────────────────────────────────────────────────────────
        "temporal": {"temperature": t_t, "humidity": t_h},
        "spatial":  {"temperature": s_t, "humidity": s_h},
        # ── File paths ────────────────────────────────────────────────────────
        "files": {
            "model_temp":      str(temp_ckpt),
            "model_hum":       str(hum_ckpt),
            "preds_temporal":  str(dirs["preds"] / f"seed{seed}_temporal_predictions.csv"),
            "preds_spatial":   str(dirs["preds"] / f"seed{seed}_spatial_predictions.csv"),
        },
    }


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    out_dir = Path(OUTPUT_DIR)

    # ── Create directory structure ────────────────────────────────────────────
    dirs = {
        "root":     out_dir,
        "preds":    ROOT / "predictions",
        "feat_imp": ROOT / "shap" / "feature_importance",
        "shap":     ROOT / "shap",
        "models":   ROOT / "models",
        "logs":     out_dir / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    import catboost
    print(f"\nCatBoost version: {catboost.__version__}")
    print(f"Seeds: {SEEDS}   Window: {WINDOW}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading data …")
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values(["Station Code", "Year", "Month"]).reset_index(drop=True)

    TARGET_COLS  = ["Temperature (Deg.Cel)", "Humidity (percent)"]
    EXCLUDE      = ["Date", "Station Code"]
    FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE]

    print(f"Using {len(FEATURE_COLS)} features (window={WINDOW} → {len(FEATURE_COLS)*WINDOW} flat)")

    # ── Compute year splits ───────────────────────────────────────────────────
    years       = sorted(df["Year"].unique())
    train_limit = years[TRAIN_YEAR_IDX]
    val_limit   = years[VAL_YEAR_IDX]

    train_df     = df[~df["Station Code"].isin(HOLDOUT_STATIONS) & (df["Year"] <= train_limit)]
    val_df       = df[~df["Station Code"].isin(HOLDOUT_STATIONS) &
                      (df["Year"] > train_limit) & (df["Year"] <= val_limit)]
    temp_test_df = df[~df["Station Code"].isin(HOLDOUT_STATIONS) & (df["Year"] > val_limit)]
    spat_test_df = df[ df["Station Code"].isin(HOLDOUT_STATIONS) & (df["Year"] > val_limit)]

    print(f"Splits: train={len(train_df):,}  val={len(val_df):,}  "
          f"t-test={len(temp_test_df):,}  s-test={len(spat_test_df):,}\n")

    # ── Save experiment config ────────────────────────────────────────────────
    experiment_config = {
        "model":             "CatBoost-Tuned",
        "data_path":         str(DATA_PATH),
        "seeds":             SEEDS,
        "window":            WINDOW,
        "n_features":        len(FEATURE_COLS),
        "feature_names":     FEATURE_COLS,
        "target_names":      TARGET_COLS,
        "holdout_stations":  HOLDOUT_STATIONS,
        "train_year_idx":    TRAIN_YEAR_IDX,
        "val_year_idx":      VAL_YEAR_IDX,
        "train_years":       [y for y in years if y <= train_limit],
        "val_years":         [y for y in years if train_limit < y <= val_limit],
        "test_years":        [y for y in years if y > val_limit],
        "rescale": {
            "formula": "raw = raw_min + (norm - 0.2) / 0.6 * (raw_max - raw_min)",
            "temperature": {"raw_min": TEMP_MIN, "raw_max": TEMP_MAX},
            "humidity":    {"raw_min": HUM_MIN,  "raw_max": HUM_MAX},
        },
        "hyperparams": {
            "fixed":       CAT_FIXED_PARAMS,
            "tuned_temp":  CAT_TUNED_PARAMS_TEMP,
            "tuned_hum":   CAT_TUNED_PARAMS_HUM,
            "iterations":  CAT_NUM_ITERATIONS,
            "early_stop":  CAT_EARLY_STOPPING,
            "note":        "Tuned params from Optuna seed=42, 50 trials",
        },
        "catboost_version": catboost.__version__,
    }
    (dirs["root"] / "experiment_config.json").write_text(
        json.dumps(experiment_config, indent=2))
    print("✓ experiment_config.json saved")

    # ── Run all seeds ─────────────────────────────────────────────────────────
    all_results      = []
    all_training_info = []

    for seed in SEEDS:
        print(f"\n{'#'*72}\n  CatBoost TUNED (26f)  Seed={seed}\n{'#'*72}")
        is_importance_seed = (seed == 42)     # feature importance + SHAP only for seed 42

        r = run_one(seed, df, FEATURE_COLS, TARGET_COLS,
                    train_df, val_df, temp_test_df, spat_test_df,
                    out_dir, dirs, is_importance_seed)
        all_results.append(r)

        # Save compact training info (separate from full metrics)
        all_training_info.append({
            "seed":                   seed,
            "train_time_min":         r["train_time_min"],
            "best_iter_temp":         r["best_iter_temp"],
            "best_iter_hum":          r["best_iter_hum"],
            "best_val_MAE_temp_norm": r["best_val_MAE_temp_norm"],
            "best_val_MAE_hum_norm":  r["best_val_MAE_hum_norm"],
            "hyperparams":            r["hyperparams"],
        })

        # Checkpoint after every seed
        (dirs["root"] / "metrics_results.json").write_text(
            json.dumps(all_results, indent=2))
        (dirs["root"] / "training_info.json").write_text(
            json.dumps(all_training_info, indent=2))

    # ── Final summary table ───────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  FINAL SUMMARY (rescaled)")
    print(f"{'='*72}")
    print(f"  {'Seed':>6}  {'Split':<10}  {'Target':<13}  {'MAE':>7}  {'RMSE':>7}  "
          f"{'R2':>7}  {'MAPE%':>7}")
    for r in all_results:
        for split in ["temporal", "spatial"]:
            for tgt, kind in [("temperature", "Temperature"), ("humidity", "Humidity")]:
                m = r[split][tgt]["rescaled"]
                print(f"  {r['seed']:>6}  {split:<10}  {kind:<13}  "
                      f"{m['MAE']:>7.4f}  {m['RMSE']:>7.4f}  "
                      f"{m['R2']:>7.4f}  {m['MAPE']:>7.4f}")
