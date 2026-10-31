"""
build_optuna_db.py

Builds optuna/catboost_optuna_study.db (SQLite Optuna storage) from the logged
trials in optuna/catboost_optuna_seed42_trials.json, so the tuning run can be
opened with the Optuna API or optuna-dashboard:

    import optuna
    study = optuna.load_study(study_name="catboost_temperature",
                              storage="sqlite:///optuna/catboost_optuna_study.db")
    study.trials_dataframe()

    optuna-dashboard sqlite:///optuna/catboost_optuna_study.db

Each stored trial carries the logged parameters, validation MAE (normalized) and
best CatBoost iteration. Trial start/end times were not logged and are left empty.
A fresh run of run_catboost_optuna.py writes this database directly.

    python scripts/build_optuna_db.py
"""

import json
from pathlib import Path

import optuna
from optuna.distributions import FloatDistribution, IntDistribution
from optuna.trial import TrialState

ROOT = Path(__file__).resolve().parent.parent
OPTUNA_DIR = ROOT / "optuna"
TRIALS_JSON = OPTUNA_DIR / "catboost_optuna_seed42_trials.json"
BEST_JSON = OPTUNA_DIR / "catboost_optuna_seed42_best_params.json"
STUDY_DB = OPTUNA_DIR / "catboost_optuna_study.db"

# Same search space as run_catboost_optuna.py (verified by verify_optuna.py)
DISTRIBUTIONS = {
    "learning_rate":       FloatDistribution(1e-3, 0.3, log=True),
    "depth":               IntDistribution(4, 10),
    "l2_leaf_reg":         FloatDistribution(1.0, 10.0, log=True),
    "bagging_temperature": FloatDistribution(0.0, 1.0),
    "random_strength":     FloatDistribution(1e-9, 10.0, log=True),
    "min_data_in_leaf":    IntDistribution(1, 100),
    "rsm":                 FloatDistribution(0.5, 1.0),
}

if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = f"sqlite:///{STUDY_DB}"
    trials = json.loads(TRIALS_JSON.read_text())
    best = json.loads(BEST_JSON.read_text())

    for target, logged in trials.items():
        name = f"catboost_{target}"
        try:
            optuna.delete_study(study_name=name, storage=storage)
        except KeyError:
            pass
        study = optuna.create_study(study_name=name, storage=storage, direction="minimize")
        study.set_user_attr("target", target)
        study.set_user_attr("objective", "validation MAE (normalized), CatBoost loss=MAE")
        study.set_user_attr("sampler", "TPESampler(seed=42)")
        study.set_user_attr("pruner", "MedianPruner (no intermediate values reported)")
        study.set_user_attr("n_trials", len(logged))
        study.set_user_attr("source", TRIALS_JSON.name)
        assert all(t["state"] == "TrialState.COMPLETE" for t in logged)
        study.add_trials([
            optuna.trial.create_trial(
                params=t["params"], distributions=DISTRIBUTIONS, value=t["value"],
                user_attrs={"best_iteration": t["best_iteration"]}, state=TrialState.COMPLETE)
            for t in logged
        ])
        # consistency check against the saved best parameters
        assert study.best_params == best[target]["best_params"], target
        assert study.best_value == best[target]["best_value"], target
        print(f"  {name:<22} {len(study.trials)} trials   best value {study.best_value:.6f}"
              f"   (trial {study.best_trial.number})")
    print(f"Saved: {STUDY_DB}")
