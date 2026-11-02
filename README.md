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
```

## Reference

Kabir MA, Chakma C (2026). HTC-LSTM-Attn in Bangladesh. *PLOS ONE*. https://doi.org/10.1371/journal.pone.0342431

## Contact

Md. Jarez Miah (jarez@iict.buet.ac.bd)
