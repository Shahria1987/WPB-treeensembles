import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

# ── CONFIG — edit these paths / options ───────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent   # repository root
SEED        = 42                                  # only seed 42 has SHAP saved
OUT_DIR     = ROOT / "figures" / "paper"            # where figures are written

SHAP_TEMP_TEMPORAL_NPZ = ROOT / "shap" / "shap_values_temp_temporal.npz"
SHAP_HUM_TEMPORAL_NPZ  = ROOT / "shap" / "shap_values_hum_temporal.npz"
SHAP_TEMP_SPATIAL_NPZ  = ROOT / "shap" / "shap_values_temp_spatial.npz"
SHAP_HUM_SPATIAL_NPZ   = ROOT / "shap" / "shap_values_hum_spatial.npz"
PRED_TEMPORAL = ROOT / "predictions" / f"seed{SEED}_temporal_predictions.csv"
PRED_SPATIAL  = ROOT / "predictions" / f"seed{SEED}_spatial_predictions.csv"

TOP_N      = 15      # number of features shown in the SHAP bar chart
TOP_HILITE = 5       # how many top features get the darker "Top 5" colour

# Rotate the SHAP feature tick labels (degrees). 45 reads well; use 90 for vertical.
SHAP_LABEL_ROT = 45

# Colour scheme (temperature = red, humidity = blue) — change freely
C_TEMP_DARK,  C_TEMP_LIGHT  = "#c0392b", "#e8a99d"
C_HUM_DARK,   C_HUM_LIGHT   = "#2c6fad", "#9cc0dd"

# Optional: prettier axis labels for SHAP features. Keys are the raw column
# names found in the data; anything not listed falls back to _prettify().
DISPLAY_NAMES = {
    # "Temp_roll_mean_3": "Temp roll.mean-3",
    # "Month_sin":        "Month (sin)",
    # ... add your own overrides here ...
}


# ── Matplotlib look-and-feel ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        120,
    "savefig.dpi":       300,
    "font.size":         12,
    "axes.titlesize":    15,
    "axes.titleweight":  "bold",
    "axes.labelsize":    13,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "legend.frameon":    True,
    "legend.framealpha": 0.9,
})


def _prettify(name: str) -> str:
    """Fallback label cleanup when a name is not in DISPLAY_NAMES."""
    return DISPLAY_NAMES.get(name, name.replace("_", " "))


# ── SHAP aggregation ──────────────────────────────────────────────────────────
def shap_contributions(npz_path):
    """
    Aggregate flat (window*F) SHAP values into a per-feature % contribution.

    mean over samples of |SHAP|  →  sum across the time-window steps
                                  →  normalise to 100 %.
    Returns a DataFrame: feature, pct (descending).
    """
    d = np.load(npz_path, allow_pickle=True)
    shap_vals     = d["shap_values"]          # (N, window*F)
    feature_names = [str(c) for c in d["feature_names"]]
    window        = int(d["window"][0])
    F             = len(feature_names)

    mean_abs = np.abs(shap_vals).mean(axis=0)          # (window*F,)
    per_feat = mean_abs.reshape(window, F).sum(axis=0) # (F,) sum across steps
    pct      = per_feat / (per_feat.sum() + 1e-12) * 100.0

    return (pd.DataFrame({"feature": feature_names, "pct": pct})
            .sort_values("pct", ascending=False)
            .reset_index(drop=True))




def _plot_shap_panel(ax, df, title, c_dark, c_light):
    """Draw a single SHAP bar-chart panel onto ax."""
    from matplotlib.patches import Patch
    top    = df.head(TOP_N).copy()
    labels = [_prettify(f) for f in top["feature"]]
    vals   = top["pct"].values
    xpos   = np.arange(len(top))
    colors = [c_dark if i < TOP_HILITE else c_light for i in range(len(top))]

    ax.bar(xpos, vals, color=colors, edgecolor="white", width=0.78)
    for x, v in zip(xpos, vals):
        ax.text(x, v + vals.max() * 0.01, f"{v:.1f}",
                ha="center", va="bottom", fontsize=9.5)

    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=SHAP_LABEL_ROT,
                       ha="right" if SHAP_LABEL_ROT not in (0, 90) else "center")
    ax.set_ylabel("Mean |SHAP| contribution (%)")
    ax.set_title(title)
    ax.set_ylim(0, vals.max() * 1.15)
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Patch(color=c_dark,  label=f"Top {TOP_HILITE}"),
                       Patch(color=c_light, label=f"Ranks {TOP_HILITE+1}-{TOP_N}")],
              loc="upper right")


def plot_shap_temporal(out_path):
    """Fig7_shap_temporal.png — Temperature (left) + Humidity (right), temporal split."""
    temp_df = shap_contributions(SHAP_TEMP_TEMPORAL_NPZ)
    hum_df  = shap_contributions(SHAP_HUM_TEMPORAL_NPZ)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    _plot_shap_panel(axes[0], temp_df, "Temperature", C_TEMP_DARK, C_TEMP_LIGHT)
    _plot_shap_panel(axes[1], hum_df,  "Humidity",    C_HUM_DARK,  C_HUM_LIGHT)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


def plot_shap_spatial(out_path):
    """Fig7_shap_spatial.png — Temperature (left) + Humidity (right), spatial split."""
    temp_df = shap_contributions(SHAP_TEMP_SPATIAL_NPZ)
    hum_df  = shap_contributions(SHAP_HUM_SPATIAL_NPZ)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    _plot_shap_panel(axes[0], temp_df, "Temperature", C_TEMP_DARK, C_TEMP_LIGHT)
    _plot_shap_panel(axes[1], hum_df,  "Humidity",    C_HUM_DARK,  C_HUM_LIGHT)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


# ── Predictions loading ───────────────────────────────────────────────────────
def load_preds():
    """Load both seed-42 prediction CSVs into one dict keyed by split."""
    return {
        "temporal": pd.read_csv(PRED_TEMPORAL),
        "spatial":  pd.read_csv(PRED_SPATIAL),
    }


# four (target, split) panels in the order used by the paper figures
PANELS = [
    ("temp", "temporal", "Temperature", "Temporal", "°C", C_TEMP_DARK),
    ("temp", "spatial",  "Temperature", "Spatial",  "°C", C_TEMP_DARK),
    ("hum",  "temporal", "Humidity",    "Temporal", "%",  C_HUM_DARK),
    ("hum",  "spatial",  "Humidity",    "Spatial",  "%",  C_HUM_DARK),
]


def plot_actual_vs_predicted(preds, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))

    for ax, (tgt, split, target_name, split_name, unit, color) in zip(
            axes.ravel(), PANELS):
        df = preds[split]
        y  = df[f"{tgt}_true_raw"].values
        p  = df[f"{tgt}_pred_raw"].values

        r2   = r2_score(y, p)
        mae  = mean_absolute_error(y, p)
        rmse = np.sqrt(mean_squared_error(y, p))

        ax.scatter(y, p, s=14, alpha=0.45, color=color, edgecolors="none")

        lo = min(y.min(), p.min())
        hi = max(y.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "--", color="0.35", lw=1.4,
                label="y = x")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")

        ax.set_title(f"{target_name} — {split_name}")
        ax.set_xlabel(f"Observed {target_name.lower()} ({unit})")
        ax.set_ylabel(f"Predicted {target_name.lower()} ({unit})")

        txt = (f"$R^2$ = {r2:.4f}\n"
               f"MAE = {mae:.3f} {unit}\n"
               f"RMSE = {rmse:.3f} {unit}")
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=11,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
        if tgt == "temp" and split == "temporal":
            ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


def plot_residuals(preds, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    for ax, (tgt, split, target_name, split_name, unit, color) in zip(
            axes.ravel(), PANELS):
        df   = preds[split]
        resid = (df[f"{tgt}_true_raw"] - df[f"{tgt}_pred_raw"]).values

        mu, sigma = resid.mean(), resid.std(ddof=1)
        skew = stats.skew(resid)
        kurt = stats.kurtosis(resid)            # excess kurtosis (Fisher)
        # Shapiro is capped at 5000 samples; subsample if needed
        sh_sample = (resid if len(resid) <= 5000
                     else np.random.default_rng(0).choice(resid, 5000, replace=False))
        shapiro_p = stats.shapiro(sh_sample).pvalue

        # histogram (density)
        ax.hist(resid, bins=40, density=True, color=color, alpha=0.45,
                label="Histogram")

        # KDE
        xs  = np.linspace(resid.min(), resid.max(), 400)
        kde = stats.gaussian_kde(resid)
        ax.plot(xs, kde(xs), color=color, lw=2.2, label="KDE")

        # normal reference
        ax.plot(xs, stats.norm.pdf(xs, mu, sigma), "--", color="0.45", lw=1.4,
                label="Normal ref.")

        # mean (dashed) and zero (dotted) reference lines
        ax.axvline(mu, color=color, ls="--", lw=1.2)
        ax.axvline(0,  color="0.4",  ls=":",  lw=1.2)

        stat_txt = (f"$\\mu$ = {mu:+.3f} {unit}\n"
                    f"$\\sigma$ = {sigma:.3f} {unit}\n"
                    f"skew = {skew:+.3f}\n"
                    f"kurt = {kurt:+.3f}\n"
                    f"Shapiro $p$ = {shapiro_p:.3f}")
        ax.text(0.03, 0.97, stat_txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=10.5,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

        ax.set_title(f"{target_name} — {split_name}")
        ax.set_xlabel(f"Residual ({unit})")
        ax.set_ylabel("Density")
        ax.legend(loc="upper right", fontsize=9.5)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading seed {SEED} outputs from: {ROOT}")

    print("\nGenerating figures …")
    plot_shap_temporal(OUT_DIR / "Fig7_shap_temporal.png")
    plot_shap_spatial(OUT_DIR  / "Fig7_shap_spatial.png")

    preds = load_preds()
    plot_actual_vs_predicted(preds, OUT_DIR / "Fig8_actual_vs_predicted.png")
    plot_residuals(preds,           OUT_DIR / "Fig9_residuals.png")

    print(f"\nDone. Figures written to: {OUT_DIR.resolve()}")
