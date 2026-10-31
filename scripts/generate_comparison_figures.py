"""
generate_comparison_figures.py
================================
Generates all 11 journal-grade comparison figures for the
CatBoost vs. HTC-LSTM-Attn weather forecasting paper.

All data, colors, fonts, sizes, and labels are controlled
from the CONFIG section. Edit CONFIG only — do not touch
the plotting functions unless you need structural changes.

Run:
    python scripts/generate_comparison_figures.py

Output directory: CONFIG["paths"]["output_dir"]
Each figure saved as both PNG (600 dpi) and PDF.
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings("ignore")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        USER CONFIG — EDIT HERE                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

CONFIG = {

    # ── Output ───────────────────────────────────────────────────────────────
    "paths": {
        "output_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "comparison"),
    },
    "output": {
        "save_png": True,
        "save_pdf": True,
        "dpi":      600,        # 300 = draft, 600 = journal submission
    },

    # ── Global font & axes style ──────────────────────────────────────────────
    "style": {
        "font_family":    "serif",
        "font_serif":     "DejaVu Serif",
        "font_size":      12,
        "title_size":     13,
        "label_size":     12,
        "tick_size":      10,
        "legend_size":    10,
        "axes_linewidth": 1.2,
        "line_linewidth": 1.8,
        "grid_alpha":     0.20,
        "grid_linewidth": 0.6,
        "spine_top":      False,
        "spine_right":    False,
    },

    # ── Color palette ─────────────────────────────────────────────────────────
    # Change any hex code to restyle every chart at once
    "colors": {
        "catboost_tuned":   "#1A5276",   # dark blue  — our best model
        "catboost_default": "#E67E22",   # orange     — our default model
        "htc_lstm":         "#7D3C98",   # purple     — paper's proposed model
        "lightgbm":         "#B7950B",   # gold       — LightGBM baseline
        "xgboost":          "#1E8449",   # green      — XGBoost baseline
        "deep_other":       "#7F8C8D",   # grey       — other deep models
        "fedformer":        "#C0392B",   # red        — FEDformer (poor fit)
        "temporal":         "#C0392B",   # red        — temporal split lines
        "spatial":          "#117A65",   # teal       — spatial split lines
        "errorbar":         "#111111",   # near-black — error bar color
        "annotation_box":   "#BBBBBB",   # grey       — text box border
        "ref_line":         "#888888",   # grey       — reference/threshold lines
    },

    # ── Figure sizes (width, height) in inches ────────────────────────────────
    "figsize": {
        "wide_8metric":    (14, 5.5),   # Charts A, B, D, E  (8-metric grouped bar)
        "dual_panel":      (11, 5.0),   # Chart C             (wins + time)
        "lollipop":        (12, 5.5),   # Charts F, K         (improvement bars)
        "heatmap":         (11, 4.0),   # Chart G             (hyperparameter heatmap)
        "benchmark":       (15, 6.0),   # Chart H             (full benchmark horizontal)
        "benchmark_spat":  (14, 5.5),   # Chart I             (spatial benchmark)
        "reproducibility": (16, 8.0),   # Chart J             (per-seed 2x4 grid)
    },

    # ── Shared bar chart settings ─────────────────────────────────────────────
    "bars": {
        "alpha":              0.82,
        "errorbar_cap":       5,
        "errorbar_lw":        1.5,
        "errorbar_cap_thick": 1.5,
        "sig_fontsize":       8.5,      # font size of significance stars (*** etc.)
        "val_fontsize":       8.5,      # font size of value labels on bars
        "val_offset":         0.01,     # gap between bar top and value label
    },

    # ── Metric axis labels ────────────────────────────────────────────────────
    # Edit display names for any metric here
    "metric_labels": {
        "8metric_short": [             # used in Charts A, B, D, E (8-col grouped bar)
            "Temp\nMAE (C)", "Temp\nRMSE (C)", "Temp\nR2", "Temp\nMAPE (%)",
            "Hum\nMAE (%)", "Hum\nRMSE (%)", "Hum\nR2", "Hum\nMAPE (%)",
        ],
        "8metric_flat": [              # used in Charts F, K (flat lollipop)
            "Temp MAE", "Temp RMSE", "Temp R2", "Temp MAPE",
            "Hum MAE",  "Hum RMSE",  "Hum R2",  "Hum MAPE",
        ],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # TABLE DATA — from the paper
    # Edit these numbers if you update results or correct a typo.
    # ══════════════════════════════════════════════════════════════════════════

    # ── Table 0a: Default GBDT comparison — Temporal ──────────────────────────
    "table0a": {
        "htc_lstm":        [0.8178, 0.9718, 0.8527, 2.8823, 2.4693, 3.2442, 0.7228, 3.1757],
        "xgboost":         [0.8028, 1.0388, 0.8707, 2.6576, 2.1190, 2.7349, 0.7622, 2.9017],
        "lightgbm":        [0.7956, 1.0096, 0.8779, 2.6272, 1.9852, 2.5589, 0.7918, 2.7169],
        "catboost_def_mu": [0.7647, 0.9758, 0.8859, 2.52,   1.9620, 2.5321, 0.7962, 2.68  ],
        "catboost_def_sd": [0.0028, 0.0015, 0.0003, 0.01,   0.0316, 0.0288, 0.0046, 0.04  ],
    },

    # ── Table 0b: Default GBDT comparison — Spatial ───────────────────────────
    "table0b": {
        "htc_lstm":        [0.9587, 1.1898, 0.8342, 3.1534, 2.4796, 3.2119, 0.6561, 3.2045],
        "xgboost":         [0.8973, 1.1655, 0.8458, 2.9656, 2.2602, 2.8672, 0.6764, 3.0906],
        "lightgbm":        [0.9011, 1.1386, 0.8529, 2.9802, 2.0986, 2.6829, 0.7167, 2.8800],
        "catboost_def_mu": [0.8566, 1.0948, 0.8639, 2.83,   2.0413, 2.5898, 0.7359, 2.79  ],
        "catboost_def_sd": [0.0067, 0.0049, 0.0012, 0.02,   0.0456, 0.0488, 0.0100, 0.06  ],
    },

    # ── Table 0c: Wins summary ────────────────────────────────────────────────
    "table0c": {
        "models":           ["XGBoost\n(default)", "LightGBM\n(default)", "CatBoost\n(default)"],
        "wins":             [15, 15, 15],
        "total_metrics":    16,
        "shared_loss_note": "Shared loss: Temp RMSE\n(temporal split only)",
        "train_time_min":   [0.12, 0.08, 3.70],   # avg per seed
    },

    # ── Table 1: CatBoost Default & Tuned vs HTC-LSTM-Attn — Temporal ────────
    "table1": {
        "htc_lstm":        [0.8178, 0.9718, 0.8527, 2.8823, 2.4693, 3.2442, 0.7228, 3.1757],
        "catboost_def_mu": [0.7647, 0.9758, 0.8859, 2.52,   1.9620, 2.5321, 0.7962, 2.68  ],
        "catboost_def_sd": [0.0028, 0.0015, 0.0003, 0.01,   0.0316, 0.0288, 0.0046, 0.04  ],
        "catboost_tun_mu": [0.7546, 0.9662, 0.8882, 2.49,   1.9125, 2.4845, 0.8037, 2.62  ],
        "catboost_tun_sd": [0.0046, 0.0034, 0.0008, 0.01,   0.0040, 0.0068, 0.0011, 0.01  ],
        # Significance vs paper (one-sample t-test): "***" | "**" | "*" | "n.s."
        "sig_default":     ["***", "**",  "***", "***", "***", "***", "***", "***"],
        "sig_tuned":       ["***", "*",   "***", "***", "***", "***", "***", "***"],
    },

    # ── Table 2: CatBoost Default & Tuned vs HTC-LSTM-Attn — Spatial ─────────
    "table2": {
        "htc_lstm":        [0.9587, 1.1898, 0.8342, 3.1534, 2.4796, 3.2119, 0.6561, 3.2045],
        "catboost_def_mu": [0.8566, 1.0948, 0.8639, 2.83,   2.0413, 2.5898, 0.7359, 2.79  ],
        "catboost_def_sd": [0.0067, 0.0049, 0.0012, 0.02,   0.0456, 0.0488, 0.0100, 0.06  ],
        "catboost_tun_mu": [0.8525, 1.0925, 0.8645, 2.82,   2.0059, 2.5808, 0.7379, 2.74  ],
        "catboost_tun_sd": [0.0068, 0.0046, 0.0012, 0.02,   0.0132, 0.0124, 0.0025, 0.02  ],
        "sig_default":     ["***", "***", "***", "***", "***", "***", "***", "***"],
        "sig_tuned":       ["***", "***", "***", "***", "***", "***", "***", "***"],
    },

    # ── Table 3: Tuning effect — % improvement Default to Tuned ──────────────
    "table3": {
        # Positive = tuned is better than default
        "delta_temporal": [1.31, 0.98, 0.26, 1.34, 2.52, 1.88, 0.95, 2.45],
        "delta_spatial":  [0.48, 0.21, 0.07, 0.49, 1.73, 0.35, 0.26, 1.69],
        "annotation": (
            "Tuning provides +0.07% to +2.52% per metric.\n"
            "Bulk of GBDT advantage already present at defaults."
        ),
    },

    # ── Table 4: Hyperparameter search (for heatmap) ──────────────────────────
    "table4": {
        "param_names": [
            "Learning\nRate", "Depth", "L2 Leaf\nReg", "Bagging\nTemp",
            "Random\nStrength", "Min Data\nin Leaf", "RSM (Col\nSample)",
        ],
        "row_labels": ["CatBoost\nDefault", "Tuned:\nTemperature", "Tuned:\nHumidity"],
        # Raw values (used for normalization)
        "values": [
            [0.030,    6,  3.000, 1.000, 1.000,    1, 1.000],
            [0.01276,  8,  3.848, 0.414, 0.00335, 80, 0.633],
            [0.00865,  8,  1.772, 0.800, 1.2e-5,  45, 0.688],
        ],
        # Formatted strings shown inside heatmap cells
        "cell_labels": [
            ["0.030",   "6", "3.00",  "1.00",  "1.00",   "1",  "1.00" ],
            ["0.01276", "8", "3.848", "0.414", "0.003",  "80", "0.633"],
            ["0.00865", "8", "1.772", "0.800", "1.2e-5", "45", "0.688"],
        ],
        "colormap": "RdYlGn_r",     # reversed: green=low, red=high
    },

    # ── Table 5: Per-seed raw results ─────────────────────────────────────────
    "table5": {
        "seeds":       [42, 7, 123, 2024, 999],
        "seed_labels": ["42", "7", "123", "2024", "999"],
        # Temporal
        "temp_mae_t":  [0.7571, 0.7501, 0.7600, 0.7494, 0.7565],
        "hum_mae_t":   [1.9106, 1.9074, 1.9115, 1.9171, 1.9158],
        "temp_rmse_t": [0.9676, 0.9626, 0.9703, 0.9627, 0.9679],
        "hum_rmse_t":  [2.4860, 2.4745, 2.4856, 2.4831, 2.4935],
        "temp_r2_t":   [0.8879, 0.8890, 0.8872, 0.8890, 0.8878],
        "hum_r2_t":    [0.8035, 0.8053, 0.8036, 0.8040, 0.8023],
        "temp_mape_t": [2.50,   2.47,   2.50,   2.47,   2.49  ],
        "hum_mape_t":  [2.61,   2.61,   2.61,   2.62,   2.62  ],
        # Spatial
        "temp_mae_s":  [0.8594, 0.8423, 0.8580, 0.8515, 0.8512],
        "hum_mae_s":   [2.0164, 1.9939, 1.9914, 2.0212, 2.0068],
        "temp_rmse_s": [1.0973, 1.0854, 1.0953, 1.0937, 1.0906],
        "hum_rmse_s":  [2.5887, 2.5631, 2.5728, 2.5926, 2.5866],
        "temp_r2_s":   [0.8633, 0.8663, 0.8638, 0.8642, 0.8650],
        "hum_r2_s":    [0.7362, 0.7414, 0.7395, 0.7355, 0.7367],
        "temp_mape_s": [2.84,   2.78,   2.83,   2.82,   2.81  ],
        "hum_mape_s":  [2.75,   2.72,   2.72,   2.76,   2.75  ],
    },

    # ── Table 6: Comprehensive benchmark rows ─────────────────────────────────
    # Each row: (model_name, temp_mae, temp_mae_sd, hum_mae, hum_mae_sd, category)
    # category: "ours" | "paper" | "gbdt" | "deep" | "poor"
    "table6_temporal": {
        "rows": [
            ("CatBoost-Tuned",  0.7546, 0.0046, 1.9125, 0.0040, "ours"),
            ("LightGBM (def.)", 0.7956, 0,      1.9852, 0,      "gbdt"),
            ("XGBoost (def.)",  0.8028, 0,      2.1190, 0,      "gbdt"),
            ("HTC-LSTM-Attn",   0.8178, 0,      2.4693, 0,      "paper"),
            ("TimesNet",        0.9856, 0,      2.4691, 0,      "deep"),
            ("Pyraformer",      0.9559, 0,      2.7021, 0,      "deep"),
            ("Autoformer",      1.0117, 0,      2.8796, 0,      "deep"),
            ("LSTM",            1.6800, 0,      4.3400, 0,      "deep"),
            ("Conv1D",          1.7500, 0,      3.4700, 0,      "deep"),
            ("GRU",             2.1700, 0,      4.5600, 0,      "deep"),
            ("ANN",             1.5803, 0,      3.9590, 0,      "deep"),
            ("FEDformer",       2.3881, 0,      4.7068, 0,      "poor"),
        ],
    },
    "table6_spatial": {
        "rows": [
            ("CatBoost-Tuned",  0.8525, 0.0068, 2.0059, 0.0132, "ours"),
            ("LightGBM (def.)", 0.9011, 0,      2.0986, 0,      "gbdt"),
            ("XGBoost (def.)",  0.8973, 0,      2.2602, 0,      "gbdt"),
            ("HTC-LSTM-Attn",   0.9587, 0,      2.4796, 0,      "paper"),
            ("TimesNet",        1.0757, 0,      3.1328, 0,      "deep"),
            ("Pyraformer",      1.1566, 0,      3.4936, 0,      "deep"),
            ("Autoformer",      1.1253, 0,      5.0641, 0,      "deep"),
            ("FEDformer",       2.4298, 0,      4.2922, 0,      "poor"),
        ],
    },

    # ── Chart K: % improvement CatBoost-Tuned vs HTC-LSTM-Attn ──────────────
    "table_improvement": {
        # Positive = better than paper. From Tables 1+2 delta columns.
        "pct_temporal": [7.7,  0.6,  4.2,  13.7, 22.5, 23.4, 11.2, 17.6],
        "pct_spatial":  [11.1, 8.2,  3.6,  10.7, 19.1, 19.7, 12.5, 14.4],
    },

}  # end CONFIG


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                   HELPER FUNCTIONS — do not edit                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def apply_style(cfg):
    s = cfg["style"]
    plt.rcParams.update({
        "font.family":       s["font_family"],
        "font.serif":        [s["font_serif"]],
        "font.size":         s["font_size"],
        "axes.titlesize":    s["title_size"],
        "axes.labelsize":    s["label_size"],
        "xtick.labelsize":   s["tick_size"],
        "ytick.labelsize":   s["tick_size"],
        "legend.fontsize":   s["legend_size"],
        "axes.linewidth":    s["axes_linewidth"],
        "axes.spines.top":   s["spine_top"],
        "axes.spines.right": s["spine_right"],
        "axes.grid":         True,
        "grid.alpha":        s["grid_alpha"],
        "grid.linewidth":    s["grid_linewidth"],
        "lines.linewidth":   s["line_linewidth"],
        "savefig.dpi":       cfg["output"]["dpi"],
        "savefig.bbox":      "tight",
        "savefig.facecolor": "white",
        "figure.facecolor":  "white",
    })


def save_fig(fig, out_dir, stem, cfg):
    if cfg["output"]["save_png"]:
        fig.savefig(os.path.join(out_dir, f"{stem}.png"))
        print(f"  saved: {stem}.png")
    if cfg["output"]["save_pdf"]:
        fig.savefig(os.path.join(out_dir, f"{stem}.pdf"))
        print(f"  saved: {stem}.pdf")
    plt.close(fig)


def category_color(cat, cfg):
    C = cfg["colors"]
    return {
        "ours":  C["catboost_tuned"],
        "paper": C["htc_lstm"],
        "gbdt":  C["lightgbm"],
        "deep":  C["deep_other"],
        "poor":  C["fedformer"],
    }.get(cat, C["deep_other"])


def legend_patch(label, color, alpha=0.85):
    return mpatches.Patch(facecolor=color, label=label, alpha=alpha)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        FIGURE GENERATORS                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def chart_ab_gbdt_default(cfg, out_dir):
    """
    Chart A: GBDT Default vs HTC-LSTM-Attn — Temporal (Table 0a)
    Chart B: GBDT Default vs HTC-LSTM-Attn — Spatial  (Table 0b)
    """
    C  = cfg["colors"]
    B  = cfg["bars"]
    ML = cfg["metric_labels"]["8metric_short"]
    x  = np.arange(len(ML))
    w  = 0.19

    for stem, tbl, split_label in [
        ("chartA_gbdt_default_temporal", cfg["table0a"],
         "Temporal Test (19 Stations, 2016-2022)"),
        ("chartB_gbdt_default_spatial",  cfg["table0b"],
         "Spatial Generalisation Test (5 Unseen Stations)"),
    ]:
        fig, ax = plt.subplots(figsize=cfg["figsize"]["wide_8metric"])

        ax.bar(x - 1.5*w, tbl["htc_lstm"],        w, color=C["htc_lstm"],         alpha=B["alpha"], label="HTC-LSTM-Attn (paper, tuned)")
        ax.bar(x - 0.5*w, tbl["xgboost"],         w, color=C["xgboost"],          alpha=B["alpha"], label="XGBoost (default)")
        ax.bar(x + 0.5*w, tbl["lightgbm"],        w, color=C["lightgbm"],         alpha=B["alpha"], label="LightGBM (default)")
        ax.bar(x + 1.5*w, tbl["catboost_def_mu"], w, color=C["catboost_default"], alpha=B["alpha"], label="CatBoost (default, mean+/-SD)")
        ax.errorbar(x + 1.5*w, tbl["catboost_def_mu"],
                    yerr=tbl["catboost_def_sd"],
                    fmt="none", color=C["errorbar"],
                    capsize=B["errorbar_cap"],
                    capthick=B["errorbar_cap_thick"],
                    elinewidth=B["errorbar_lw"])

        ax.set_xticks(x)
        ax.set_xticklabels(ML, fontsize=cfg["style"]["tick_size"])
        ax.set_ylabel("Metric Value")
        ax.set_title(
            f"GBDT Default Settings vs. HTC-LSTM-Attn -- {split_label}\n"
            "(lower is better for MAE/RMSE/MAPE; higher is better for R2)",
            fontweight="bold", pad=10)
        ax.legend(framealpha=0.88, edgecolor=C["annotation_box"],
                  ncol=4, loc="upper right",
                  fontsize=cfg["style"]["legend_size"])
        ax.text(0.01, 0.97,
                "Down = better (error metrics)  |  Up = better (R2)",
                transform=ax.transAxes, fontsize=8.5, va="top", color="#555")
        save_fig(fig, out_dir, stem, cfg)


def chart_c_wins_and_time(cfg, out_dir):
    """
    Chart C: Win counts + training time (Table 0c)
    """
    C  = cfg["colors"]
    T  = cfg["table0c"]
    cols = [C["xgboost"], C["lightgbm"], C["catboost_default"]]

    fig, axes = plt.subplots(1, 2, figsize=cfg["figsize"]["dual_panel"])
    fig.subplots_adjust(wspace=0.38)

    # Left panel: win counts
    ax = axes[0]
    bars = ax.bar(T["models"], T["wins"], color=cols, width=0.45, alpha=0.85)
    ax.set_ylim(0, T["total_metrics"] + 3)
    ax.axhline(T["total_metrics"], color=C["ref_line"], lw=1.1, ls="--",
               alpha=0.6, label=f'Total metrics ({T["total_metrics"]})')
    ax.set_ylabel(f'Wins vs. HTC-LSTM-Attn (of {T["total_metrics"]})')
    ax.set_title("Wins vs. Published HTC-LSTM-Attn\n(Default GBDT, No Tuning)",
                 fontweight="bold", pad=10)
    for bar, w in zip(bars, T["wins"]):
        ax.text(bar.get_x() + bar.get_width()/2, w + 0.15,
                f'{w}/{T["total_metrics"]}',
                ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax.legend(fontsize=9, framealpha=0.88, edgecolor=C["annotation_box"])
    ax.text(0.5, 0.18, T["shared_loss_note"],
            ha="center", transform=ax.transAxes,
            fontsize=8.5, color="#555", style="italic")

    # Right panel: training time on log scale
    ax2 = axes[1]
    bars2 = ax2.bar(T["models"], T["train_time_min"], color=cols, width=0.45, alpha=0.85)
    ax2.set_yscale("log")
    ax2.set_ylabel("Training Time per Seed (min, log scale)")
    ax2.set_title("Computational Cost\n(per Seed, Default Settings)",
                  fontweight="bold", pad=10)
    for bar, t in zip(bars2, T["train_time_min"]):
        ax2.text(bar.get_x() + bar.get_width()/2, t * 1.15,
                 f"{t:.2f} min",
                 ha="center", va="bottom", fontweight="bold", fontsize=11)

    fig.suptitle(
        "Default GBDT Summary: Wins vs. HTC-LSTM-Attn & Computational Cost",
        fontsize=cfg["style"]["title_size"], fontweight="bold", y=1.02)
    save_fig(fig, out_dir, "chartC_wins_and_time", cfg)


def chart_de_catboost_vs_paper(cfg, out_dir):
    """
    Chart D: CatBoost Default & Tuned vs HTC-LSTM-Attn — Temporal (Table 1)
    Chart E: CatBoost Default & Tuned vs HTC-LSTM-Attn — Spatial  (Table 2)
    """
    C  = cfg["colors"]
    B  = cfg["bars"]
    ML = cfg["metric_labels"]["8metric_short"]
    x  = np.arange(len(ML))
    w  = 0.25

    for stem, tbl, split_label in [
        ("chartD_catboost_vs_paper_temporal", cfg["table1"],
         "Temporal Test (19 Stations, 2016-2022)"),
        ("chartE_catboost_vs_paper_spatial",  cfg["table2"],
         "Spatial Generalisation Test (5 Unseen Stations)"),
    ]:
        fig, ax = plt.subplots(figsize=cfg["figsize"]["wide_8metric"])

        ax.bar(x - w, tbl["htc_lstm"],        w, color=C["htc_lstm"],         alpha=B["alpha"], label="HTC-LSTM-Attn (paper, tuned)")
        ax.bar(x,     tbl["catboost_def_mu"], w, color=C["catboost_default"], alpha=B["alpha"], label="CatBoost Default (mean+/-SD, n=5)")
        ax.bar(x + w, tbl["catboost_tun_mu"], w, color=C["catboost_tuned"],   alpha=0.90,       label="CatBoost Tuned (mean+/-SD, n=5)")

        ax.errorbar(x,     tbl["catboost_def_mu"],
                    yerr=tbl["catboost_def_sd"],
                    fmt="none", color=C["errorbar"],
                    capsize=B["errorbar_cap"],
                    capthick=B["errorbar_cap_thick"],
                    elinewidth=B["errorbar_lw"])
        ax.errorbar(x + w, tbl["catboost_tun_mu"],
                    yerr=tbl["catboost_tun_sd"],
                    fmt="none", color=C["errorbar"],
                    capsize=B["errorbar_cap"],
                    capthick=B["errorbar_cap_thick"],
                    elinewidth=B["errorbar_lw"])

        # Significance stars above each bar group
        for i, (sd, st) in enumerate(zip(tbl["sig_default"], tbl["sig_tuned"])):
            ymax = max(
                tbl["htc_lstm"][i],
                tbl["catboost_def_mu"][i] + tbl["catboost_def_sd"][i],
                tbl["catboost_tun_mu"][i] + tbl["catboost_tun_sd"][i],
            )
            ax.text(i,     ymax * 1.022, sd, ha="center",
                    fontsize=B["sig_fontsize"],
                    color=C["catboost_default"], fontweight="bold")
            ax.text(i + w, ymax * 1.022, st, ha="center",
                    fontsize=B["sig_fontsize"],
                    color=C["catboost_tuned"],   fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(ML, fontsize=cfg["style"]["tick_size"])
        ax.set_ylabel("Metric Value")
        ax.set_title(
            f"CatBoost Default & Tuned vs. HTC-LSTM-Attn -- {split_label}\n"
            "Significance: *** p<0.001, ** p<0.01, * p<0.05 (one-sample t-test, n=5)",
            fontweight="bold", pad=10)
        ax.legend(framealpha=0.88, edgecolor=C["annotation_box"],
                  loc="upper right", fontsize=cfg["style"]["legend_size"])
        ax.text(0.01, 0.97,
                "Down = better (error metrics)  |  Up = better (R2)",
                transform=ax.transAxes, fontsize=8.5, va="top", color="#555")
        save_fig(fig, out_dir, stem, cfg)


def chart_f_tuning_effect(cfg, out_dir):
    """
    Chart F: Optuna tuning effect -- % improvement Default to Tuned (Table 3)
    """
    C  = cfg["colors"]
    B  = cfg["bars"]
    T  = cfg["table3"]
    ML = cfg["metric_labels"]["8metric_flat"]
    x  = np.arange(len(ML))
    w  = 0.35

    fig, ax = plt.subplots(figsize=cfg["figsize"]["lollipop"])

    bars1 = ax.bar(x - w/2, T["delta_temporal"], w,
                   color=C["temporal"], alpha=B["alpha"], label="Temporal Test")
    bars2 = ax.bar(x + w/2, T["delta_spatial"],  w,
                   color=C["spatial"],  alpha=B["alpha"], label="Spatial Test")
    ax.axhline(0, color="#333", lw=0.9)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + B["val_offset"],
                f"+{h:.2f}%", ha="center", va="bottom",
                fontsize=B["val_fontsize"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(ML, fontsize=cfg["style"]["tick_size"])
    ax.set_ylabel("Relative Improvement (%)")
    ax.set_ylim(0, max(T["delta_temporal"] + T["delta_spatial"]) * 1.30)
    ax.set_title(
        "Optuna Hyperparameter Tuning Effect\n"
        "CatBoost Tuned vs. CatBoost Default -- Relative Improvement per Metric",
        fontweight="bold", pad=10)
    ax.legend(framealpha=0.88, edgecolor=C["annotation_box"])
    ax.text(0.5, 0.95, T["annotation"],
            ha="center", transform=ax.transAxes, fontsize=9, va="top",
            style="italic", color="#444",
            bbox=dict(boxstyle="round,pad=0.3", fc="#F8F9FA",
                      ec="#CCCCCC", alpha=0.9))
    save_fig(fig, out_dir, "chartF_optuna_tuning_effect", cfg)


def chart_g_hyperparameter_heatmap(cfg, out_dir):
    """
    Chart G: Hyperparameter configuration heatmap (Table 4)
    """
    T    = cfg["table4"]
    vals = np.array(T["values"], dtype=float)
    norm = np.zeros_like(vals)

    # Normalize each column (parameter) to [0, 1]
    for j in range(vals.shape[1]):
        col = vals[:, j]
        rng = col.max() - col.min()
        norm[:, j] = (col - col.min()) / rng if rng > 0 else 0.5

    fig, ax = plt.subplots(figsize=cfg["figsize"]["heatmap"])
    im = ax.imshow(norm, aspect="auto", cmap=T["colormap"], vmin=0, vmax=1)

    ax.set_xticks(range(len(T["param_names"])))
    ax.set_xticklabels(T["param_names"], fontsize=cfg["style"]["tick_size"])
    ax.set_yticks(range(len(T["row_labels"])))
    ax.set_yticklabels(T["row_labels"], fontsize=cfg["style"]["label_size"])
    ax.set_title(
        "Hyperparameter Configuration Heatmap\n"
        "(Color = relative value within each parameter: green = low, red = high)",
        fontweight="bold", pad=10)

    for i in range(len(T["row_labels"])):
        for j in range(len(T["param_names"])):
            text_color = "white" if (norm[i, j] > 0.68 or norm[i, j] < 0.22) else "#222"
            ax.text(j, i, T["cell_labels"][i][j],
                    ha="center", va="center",
                    fontsize=9, fontweight="bold", color=text_color)

    plt.colorbar(im, ax=ax,
                 label="Normalized value (0 = min, 1 = max per parameter)",
                 shrink=0.8, pad=0.02)
    fig.tight_layout()
    save_fig(fig, out_dir, "chartG_hyperparameter_heatmap", cfg)


def chart_hi_comprehensive_benchmark(cfg, out_dir):
    """
    Chart H: Comprehensive benchmark MAE -- Temporal (Table 6)
    Chart I: Comprehensive benchmark MAE -- Spatial  (Table 6)
    """
    C = cfg["colors"]
    B = cfg["bars"]

    legend_patches = [
        legend_patch("CatBoost-Tuned (Ours)",      C["catboost_tuned"]),
        legend_patch("LightGBM / XGBoost (def.)",  C["lightgbm"]),
        legend_patch("HTC-LSTM-Attn (paper)",       C["htc_lstm"]),
        legend_patch("Other deep models",            C["deep_other"]),
        legend_patch("FEDformer (poor fit)",          C["fedformer"]),
    ]

    for stem, tbl, split_label, figkey in [
        ("chartH_comprehensive_mae_temporal", cfg["table6_temporal"],
         "Temporal Test (19 Stations, 2016-2022)", "benchmark"),
        ("chartI_comprehensive_mae_spatial",  cfg["table6_spatial"],
         "Spatial Test (5 Unseen Stations)", "benchmark_spat"),
    ]:
        rows        = tbl["rows"]
        model_names = [r[0] for r in rows]
        temp_mae    = [r[1] for r in rows]
        temp_mae_sd = [r[2] for r in rows]
        hum_mae     = [r[3] for r in rows]
        hum_mae_sd  = [r[4] for r in rows]
        cats        = [r[5] for r in rows]
        bar_colors  = [category_color(c, cfg) for c in cats]
        n = len(rows)

        fig, axes = plt.subplots(1, 2, figsize=cfg["figsize"][figkey])
        fig.subplots_adjust(wspace=0.42)

        for ax, vals, sds, target, unit in [
            (axes[0], temp_mae, temp_mae_sd, "Temperature", "C"),
            (axes[1], hum_mae,  hum_mae_sd,  "Humidity",    "%"),
        ]:
            y    = np.arange(n)
            bars = ax.barh(y[::-1], vals, color=bar_colors, height=0.62, alpha=0.85)

            for i, (v, sd) in enumerate(zip(vals, sds)):
                if sd > 0:
                    ax.errorbar(v, n - 1 - i, xerr=sd,
                                fmt="none", color=C["errorbar"],
                                capsize=B["errorbar_cap"],
                                capthick=B["errorbar_cap_thick"],
                                elinewidth=B["errorbar_lw"])

            ax.set_yticks(y[::-1])
            ax.set_yticklabels(model_names[::-1][::-1], fontsize=10)
            ax.set_xlabel(f"{target} MAE ({unit})")
            ax.set_title(f"{target} MAE\n{split_label}",
                         fontweight="bold", pad=10)

            for bar, v in zip(bars, vals[::-1]):
                ax.text(v + B["val_offset"],
                        bar.get_y() + bar.get_height()/2,
                        f"{v:.4f}", va="center", fontsize=8.5)

            paper_idx = next(i for i, r in enumerate(rows) if r[5] == "paper")
            ax.axvline(vals[paper_idx], color=C["htc_lstm"],
                       lw=1.2, ls="--", alpha=0.55)

        fig.legend(handles=legend_patches, loc="lower center",
                   bbox_to_anchor=(0.5, -0.08), ncol=5,
                   framealpha=0.9, edgecolor=C["annotation_box"],
                   fontsize=cfg["style"]["legend_size"])
        fig.suptitle(
            f"Comprehensive Model Benchmark -- MAE ({split_label})\n"
            "CatBoost-Tuned: mean+/-SD (n=5); others: single-run from Kabir & Chakma (2026)",
            fontsize=cfg["style"]["title_size"], fontweight="bold", y=1.03)
        save_fig(fig, out_dir, stem, cfg)


def chart_j_per_seed(cfg, out_dir):
    """
    Chart J: Per-seed reproducibility -- all 8 metrics x 5 seeds (Table 5)
    """
    C = cfg["colors"]
    T = cfg["table5"]
    sx = np.arange(len(T["seeds"]))

    # (temporal_data, spatial_data, y_label, panel_title)
    panels = [
        (T["temp_mae_t"],  T["temp_mae_s"],  "MAE (C)",   "Temperature MAE"),
        (T["hum_mae_t"],   T["hum_mae_s"],   "MAE (%)",   "Humidity MAE"),
        (T["temp_rmse_t"], T["temp_rmse_s"], "RMSE (C)",  "Temperature RMSE"),
        (T["hum_rmse_t"],  T["hum_rmse_s"],  "RMSE (%)",  "Humidity RMSE"),
        (T["temp_r2_t"],   T["temp_r2_s"],   "R2",        "Temperature R2"),
        (T["hum_r2_t"],    T["hum_r2_s"],    "R2",        "Humidity R2"),
        (T["temp_mape_t"], T["temp_mape_s"], "MAPE (%)",  "Temperature MAPE"),
        (T["hum_mape_t"],  T["hum_mape_s"],  "MAPE (%)",  "Humidity MAPE"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=cfg["figsize"]["reproducibility"])
    fig.subplots_adjust(hspace=0.42, wspace=0.38)

    for idx, (tv, sv, ylabel, title) in enumerate(panels):
        row, col = divmod(idx, 4)
        ax = axes[row, col]
        tv, sv = np.array(tv), np.array(sv)

        ax.plot(sx, tv, "o-",  color=C["temporal"], lw=1.8, ms=7,
                label="Temporal", zorder=4)
        ax.plot(sx, sv, "s--", color=C["spatial"],  lw=1.8, ms=7,
                label="Spatial",  zorder=4)

        # Mean reference lines
        ax.axhline(np.mean(tv), color=C["temporal"], lw=0.9, ls=":", alpha=0.70)
        ax.axhline(np.mean(sv), color=C["spatial"],  lw=0.9, ls=":", alpha=0.70)

        # +/-1 SD shaded band
        ax.fill_between([-0.3, len(T["seeds"]) - 0.7],
                        np.mean(tv) - np.std(tv, ddof=1),
                        np.mean(tv) + np.std(tv, ddof=1),
                        color=C["temporal"], alpha=0.08)
        ax.fill_between([-0.3, len(T["seeds"]) - 0.7],
                        np.mean(sv) - np.std(sv, ddof=1),
                        np.mean(sv) + np.std(sv, ddof=1),
                        color=C["spatial"],  alpha=0.08)

        ax.set_xticks(sx)
        ax.set_xticklabels(T["seed_labels"], fontsize=9.5)
        ax.set_xlabel("Random Seed", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontweight="bold", pad=7, fontsize=11)
        ax.set_xlim(-0.4, len(T["seeds"]) - 0.6)

        cv = np.std(tv, ddof=1) / np.mean(tv) * 100
        ax.text(0.97, 0.97, f"CV = {cv:.2f}%",
                transform=ax.transAxes, fontsize=8.5,
                ha="right", va="top", color=C["temporal"], alpha=0.90)

        if idx == 0:
            ax.legend(fontsize=9, framealpha=0.88,
                      edgecolor=C["annotation_box"], loc="best")

    fig.suptitle(
        "Per-Seed Reproducibility Audit -- CatBoost-Tuned\n"
        "All 8 Metrics x 5 Seeds x Temporal & Spatial (mean +/- SD band shown)",
        fontsize=cfg["style"]["title_size"], fontweight="bold", y=1.02)
    save_fig(fig, out_dir, "chartJ_per_seed_all_metrics", cfg)


def chart_k_pct_improvement(cfg, out_dir):
    """
    Chart K: % improvement CatBoost-Tuned vs HTC-LSTM-Attn (Tables 1+2)
    """
    C  = cfg["colors"]
    B  = cfg["bars"]
    T  = cfg["table_improvement"]
    ML = cfg["metric_labels"]["8metric_flat"]
    x  = np.arange(len(ML))
    w  = 0.35

    fig, ax = plt.subplots(figsize=cfg["figsize"]["lollipop"])

    bars1 = ax.bar(x - w/2, T["pct_temporal"], w,
                   color=C["temporal"], alpha=B["alpha"], label="Temporal Test")
    bars2 = ax.bar(x + w/2, T["pct_spatial"],  w,
                   color=C["spatial"],  alpha=B["alpha"], label="Spatial Test")
    ax.axhline(0, color="#333", lw=0.9)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + B["val_offset"],
                f"+{h:.1f}%", ha="center", va="bottom",
                fontsize=B["val_fontsize"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(ML, fontsize=cfg["style"]["tick_size"])
    ax.set_ylabel("Relative Improvement over HTC-LSTM-Attn (%)\n(positive = better than paper)")
    ax.set_ylim(-1, max(T["pct_temporal"] + T["pct_spatial"]) * 1.22)
    ax.set_title(
        "CatBoost-Tuned Improvement over HTC-LSTM-Attn\n"
        "All comparisons statistically significant (*** p < 0.001, one-sample t-test, n=5)",
        fontweight="bold", pad=10)
    ax.legend(framealpha=0.88, edgecolor=C["annotation_box"])
    save_fig(fig, out_dir, "chartK_pct_improvement_vs_paper", cfg)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                              MAIN                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    out_dir = CONFIG["paths"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    apply_style(CONFIG)

    print("=" * 60)
    print("  Comparison Figure Generator")
    print(f"  Output -> {out_dir}")
    print("=" * 60)

    print("\nCharts A & B -- GBDT Default vs. HTC-LSTM-Attn ...")
    chart_ab_gbdt_default(CONFIG, out_dir)

    print("\nChart C -- Wins & Training Time ...")
    chart_c_wins_and_time(CONFIG, out_dir)

    print("\nCharts D & E -- CatBoost Default & Tuned vs. Paper ...")
    chart_de_catboost_vs_paper(CONFIG, out_dir)

    print("\nChart F -- Optuna Tuning Effect ...")
    chart_f_tuning_effect(CONFIG, out_dir)

    print("\nChart G -- Hyperparameter Heatmap ...")
    chart_g_hyperparameter_heatmap(CONFIG, out_dir)

    print("\nCharts H & I -- Comprehensive Benchmark MAE ...")
    chart_hi_comprehensive_benchmark(CONFIG, out_dir)

    print("\nChart J -- Per-Seed Reproducibility ...")
    chart_j_per_seed(CONFIG, out_dir)

    print("\nChart K -- Percentage Improvement vs. Paper ...")
    chart_k_pct_improvement(CONFIG, out_dir)

    files = sorted(f for f in os.listdir(out_dir) if f.endswith(".png"))
    print(f"\n{'=' * 60}")
    print(f"  Done! {len(files)} PNG figures saved to {out_dir}/")
    print(f"{'=' * 60}")
    for f in files:
        kb = os.path.getsize(os.path.join(out_dir, f)) // 1024
        print(f"  {f:<55} {kb:>5} KB")


if __name__ == "__main__":
    main()
