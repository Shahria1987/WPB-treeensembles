"""
run_catboost_optuna.py

CatBoost with 26 features, hyperparameters tuned with Optuna
(TPE sampler, seed=42, 50 trials per target). Temperature and humidity are
tuned separately on the validation split (2013-2015), then refit with the
best parameters and evaluated on the temporal and spatial test splits.

Outputs (in OUTPUT_DIR):
    catboost_optuna_seed42.log               full run log
    catboost_optuna_seed42_best_params.json  best parameters + best validation MAE
    catboost_optuna_seed42_trials.json       every trial (params, value, best iteration)
    catboost_optuna_results.json             test metrics (normalized + rescaled)
    catboost_optuna_study.db                 Optuna studies (SQLite), one per target

Reconstruction note
-------------------
The original copy of this script was lost; this file was rebuilt from the
run outputs in ../optuna and the shared pipeline of the
other 26-feature scripts in this repository. The search space and sampler
settings are confirmed exactly: replaying the logged objective values through
this search space with TPESampler(seed=42) reproduces all 50 trial parameter
sets for both targets (relative difference < 1e-14; see
verify_optuna.py). Use optuna < 5.0 (the runs used Optuna 4.x);
Optuna 5.0 changed the TPE sampler and gives different trials after the first 10.
"""

import sys, time, random, json, logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Config ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent   # repository root
DATA_PATH  = ROOT / "data" / "Preprocessed_Weather_Data_final.csv"
# On Kaggle: DATA_PATH = "/kaggle/input/datasets/shahriahasan/weather-forecast-bd/Preprocessed_Weather_Data_final.csv"
OUTPUT_DIR = ROOT / "optuna"
STUDY_DB   = OUTPUT_DIR / "catboost_optuna_study.db"   # SQLite Optuna storage

SEED     = 42
WINDOW   = 12
N_TRIALS = 50

HOLDOUT_STATIONS = [10120, 10724, 11513, 11805, 12007]
TRAIN_YEAR_IDX = 51   # → 2012
VAL_YEAR_IDX   = 54   # → 2015

TEMP_MIN, TEMP_MAX = 16.95, 39.31
HUM_MIN,  HUM_MAX  = 39.81, 90.06

# Fixed CatBoost settings (same objective as the default baseline)
CAT_FIXED_PARAMS = {
    "loss_function": "MAE",
    "eval_metric": "MAE",
    "verbose": 0,
    "allow_writing_files": False,
    "thread_count": -1,
}
CAT_NUM_ITERATIONS  = 2000
CAT_EARLY_STOPPING  = 50


# ── Search space ────────────────────────────────────────────────────────────
def suggest_params(trial):
    return {
        "learning_rate":       trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "depth":               trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength":     trial.suggest_float("random_strength", 1e-9, 10.0, log=True),
        "min_data_in_leaf":    trial.suggest_int("min_data_in_leaf", 1, 100),
        "rsm":                 trial.suggest_float("rsm", 0.5, 1.0),
    }


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
    return raw_min + (norm - 0.2) / 0.6 * (raw_max - raw_min)


# ── Metrics ─────────────────────────────────────────────────────────────────
def compute_metrics(y_true_raw, y_pred_raw):
    return {
        "MAE":  round(float(mean_absolute_error(y_true_raw, y_pred_raw)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))), 4),
        "R2":   round(float(r2_score(y_true_raw, y_pred_raw)), 4),
        "MAPE": round(float(np.mean(np.abs((y_true_raw - y_pred_raw) /
                                           (np.abs(y_true_raw) + 1e-10))) * 100), 4),
    }


def evaluate_predictions(y_norm, p_norm, kind):
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


# ── CatBoost fit (used by every trial and by the final refit) ───────────────
def fit_cat(params, X_tr, y_tr, X_vl, y_vl, seed):
    full = dict(CAT_FIXED_PARAMS)
    full.update(params)
    full["random_seed"] = seed
    full["iterations"] = CAT_NUM_ITERATIONS
    full["early_stopping_rounds"] = CAT_EARLY_STOPPING
    model = CatBoostRegressor(**full)
    model.fit(X_tr, y_tr, eval_set=(X_vl, y_vl), verbose=False)
    best_iter = int(model.best_iteration_)
    best_val = float(model.best_score_["validation"]["MAE"])
    return model, best_iter, best_val


# ── Optuna tuning for one target ────────────────────────────────────────────
def tune_target(X_tr, y_tr, X_vl, y_vl, seed, logger, target_name):
    logger.info(f"\n{'─'*72}")
    logger.info(f"  Optuna tuning for {target_name} ({N_TRIALS} trials, seed={seed})")
    logger.info(f"{'─'*72}")

    def objective(trial):
        params = suggest_params(trial)
        _, best_iter, best_val = fit_cat(params, X_tr, y_tr, X_vl, y_vl, seed)
        trial.set_user_attr("best_iteration", best_iter)
        return best_val

    t0 = time.time()

    def log_trial(study, trial):
        n = trial.number + 1
        elapsed = time.time() - t0
        eta = elapsed / n * (N_TRIALS - n)
        logger.info(
            f"  [{target_name}] Trial {n:>3}/{N_TRIALS}  "
            f"val={trial.value:<12.6f}  best={study.best_value:<10.6f}  "
            f"best_iter={trial.user_attrs.get('best_iteration', -1):<4}  "
            f"state={trial.state.name}  elapsed={elapsed:>5.0f}s  ETA={eta:>5.0f}s"
        )

    # No intermediate values are reported, so the median pruner never prunes
    # a trial (all 100 logged trials are COMPLETE).
    storage = f"sqlite:///{STUDY_DB}"
    study_name = f"catboost_{target_name.lower()}"
    try:                                   # re-running replaces the stored study
        optuna.delete_study(study_name=study_name, storage=storage)
    except KeyError:
        pass
    study = optuna.create_study(direction="minimize",
                                sampler=TPESampler(seed=seed),
                                pruner=MedianPruner(),
                                storage=storage, study_name=study_name)
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[log_trial])
    tune_min = (time.time() - t0) / 60

    logger.info(f"  → tuning time: {tune_min:.2f} min  "
                f"best val MAE (norm): {study.best_value:.6f}")
    logger.info(f"  → best params: {study.best_params}")

    trials = [{
        "number": t.number,
        "value": t.value,
        "params": t.params,
        "state": str(t.state),
        "best_iteration": t.user_attrs.get("best_iteration"),
    } for t in study.trials]
    return study, trials, tune_min


# ── Single seed run ─────────────────────────────────────────────────────────
def run_one(seed, df, feature_cols, target_cols,
            train_df, val_df, temp_test_df, spat_test_df, out_dir):
    set_seed(seed)
    tag = f"catboost_optuna_seed{seed}"
    logger = make_logger(tag, out_dir / f"{tag}.log")

    logger.info(f"{'='*72}")
    logger.info(f"  CatBoost + Optuna (26 features)  —  seed={seed}  window={WINDOW}")
    logger.info(f"  n_trials={N_TRIALS}  sampler=TPE(seed={seed})  pruner=Median")
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

    t_start = time.time()
    st_t, trials_t, tune_t = tune_target(X_tr, y_tr[:, 0], X_vl, y_vl[:, 0],
                                         seed, logger, "Temperature")
    st_h, trials_h, tune_h = tune_target(X_tr, y_tr[:, 1], X_vl, y_vl[:, 1],
                                         seed, logger, "Humidity")

    # Save tuning outputs
    (out_dir / f"{tag}_trials.json").write_text(
        json.dumps({"temperature": trials_t, "humidity": trials_h}, indent=2))
    (out_dir / f"{tag}_best_params.json").write_text(json.dumps({
        "temperature": {"best_params": st_t.best_params, "best_value": st_t.best_value},
        "humidity":    {"best_params": st_h.best_params, "best_value": st_h.best_value},
    }, indent=2))

    # Refit with the best parameters
    logger.info(f"\n  Refitting CatBoost for Temperature with best params …")
    m_t, bi_t, bv_t = fit_cat(st_t.best_params, X_tr, y_tr[:, 0], X_vl, y_vl[:, 0], seed)
    logger.info(f"  → best iter: {bi_t}   best val MAE (norm): {bv_t:.6f}")

    logger.info(f"\n  Refitting CatBoost for Humidity with best params …")
    m_h, bi_h, bv_h = fit_cat(st_h.best_params, X_tr, y_tr[:, 1], X_vl, y_vl[:, 1], seed)
    logger.info(f"  → best iter: {bi_h}   best val MAE (norm): {bv_h:.6f}")

    total_min = (time.time() - t_start) / 60
    logger.info(f"\n  Total time (tune + refit): {total_min:.2f} min")

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
        "tag": tag, "model": "CatBoost-Optuna", "seed": seed, "window": WINDOW,
        "n_features": len(feature_cols), "n_flat_features": len(feature_cols) * WINDOW,
        "n_trials": N_TRIALS,
        "best_iter_temp": bi_t, "best_iter_hum": bi_h,
        "best_val_temp": bv_t, "best_val_hum": bv_h,
        "best_params_temp": st_t.best_params, "best_params_hum": st_h.best_params,
        "tune_time_temp_min": round(tune_t, 2), "tune_time_hum_min": round(tune_h, 2),
        "total_time_min": round(total_min, 2),
        "rescale": {"temp": [TEMP_MIN, TEMP_MAX], "hum": [HUM_MIN, HUM_MAX]},
        "temporal": {"temperature": t_t, "humidity": t_h},
        "spatial":  {"temperature": s_t, "humidity": s_h},
    }


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    out_dir = Path(OUTPUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df = df.sort_values(["Station Code", "Year", "Month"]).reset_index(drop=True)

    TARGET_COLS = ["Temperature (Deg.Cel)", "Humidity (percent)"]
    EXCLUDE     = ["Date", "Station Code"]
    FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE]

    years = sorted(df["Year"].unique())
    train_limit = years[TRAIN_YEAR_IDX]
    val_limit   = years[VAL_YEAR_IDX]

    train_df     = df[~df["Station Code"].isin(HOLDOUT_STATIONS) & (df["Year"] <= train_limit)]
    val_df       = df[~df["Station Code"].isin(HOLDOUT_STATIONS) &
                      (df["Year"] > train_limit) & (df["Year"] <= val_limit)]
    temp_test_df = df[~df["Station Code"].isin(HOLDOUT_STATIONS) & (df["Year"] > val_limit)]
    spat_test_df = df[ df["Station Code"].isin(HOLDOUT_STATIONS) & (df["Year"] > val_limit)]

    r = run_one(SEED, df, FEATURE_COLS, TARGET_COLS,
                train_df, val_df, temp_test_df, spat_test_df, out_dir)
    (out_dir / "catboost_optuna_results.json").write_text(json.dumps([r], indent=2))

    print(f"\nDone. Results saved to: {OUTPUT_DIR}")
