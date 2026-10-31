# WPB-treeensembles

Code, models, predictions, SHAP results and Optuna files for:

**Simpler, Faster, Stronger? Benchmarking Tree Ensembles Against Deep Learning for Monthly Weather Forecasting in Bangladesh**
Shahria Hasan and Md. Jarez Miah, IICT, BUET.

## Structure

```
data/          dataset link (DOI 10.6084/m9.figshare.29876786)
scripts/       all code
models/        CatBoost-Tuned models (seeds 42, 7, 123, 2024, 999; h2_h3/ for 2- and 3-month horizons)
predictions/   test predictions and bootstrap CIs
shap/          SHAP values and feature importance
optuna/        Optuna study database and best parameters
results/       metrics and logs
figures/       paper figures
```

## Reproduce

```bash
pip install -r requirements.txt
# place the dataset in data/, then run from the repository root:

python scripts/run_lightgbm_default.py
python scripts/run_xgboost_default.py
python scripts/run_catboost_default.py
python scripts/run_catboost_optuna.py      # ~3.7 h on CPU
python scripts/run_catboost_tuned.py
python scripts/block_bootstrap_ci.py
python scripts/run_catboost_h2_h3.py
python scripts/generate_paper_figures.py
python scripts/generate_comparison_figures.py
```

## Optuna

`optuna/catboost_optuna_study.db` contains the studies `catboost_temperature` and `catboost_humidity` (50 trials each). It was rebuilt from the logged trials with `build_optuna_db.py`. `run_catboost_optuna.py` was reconstructed after the original was lost, and `verify_optuna.py` confirms it reproduces all 50 logged trials. Use `optuna<5`.

## Reference

Kabir MA, Chakma C (2026). HTC-LSTM-Attn in Bangladesh. *PLOS ONE*. https://doi.org/10.1371/journal.pone.0342431

## Contact

Md. Jarez Miah (jarez@iict.buet.ac.bd)
