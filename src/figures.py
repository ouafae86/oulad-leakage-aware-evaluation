"""
figures.py
─────────────────────────────────────────────────────────────────
Genere l'ensemble des figures et tableaux visuels destines a
l'article, a partir des resultats deja calcules dans interim/.
Ce module NE RECALCULE RIEN — il lit uniquement les sorties des
modules precedents (ablation_results.csv, baselines_results.csv,
shap_*, embeddings.npy, recon_mse.npy, autoencoder_history.json).

Figures produites (dans ../figures/) :
  fig1_autoencoder_convergence.png   — courbe de perte (train/val)
  fig2_ablation_performance.png      — AUC des 9 combinaisons config x modele
  fig3_baselines_vs_ours.png         — comparaison B1/B2/B3 vs notre meilleure config
  fig4_shap_top_features.png         — importance SHAP (bar chart)
  fig5_shap_summary_beeswarm.png     — SHAP summary plot (essaim de points)
  fig6_confusion_matrix.png          — matrice de confusion (meilleure config, out-of-fold)
  fig7_reconstruction_by_class.png   — verification qualitative (erreur AE par classe)
  table1_ablation.png                — tableau recapitulatif (image, pret a inserer)

Usage :
    python3 figures.py
─────────────────────────────────────────────────────────────────
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import confusion_matrix

from features_aggregated import AGGREGATED_FEATURE_NAMES

INTERIM_DIR = os.path.join(os.path.dirname(__file__), "..", "interim")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RANDOM_SEED = 42

BLUE, GREEN, AMBER, RED, GRAY, LIGHT = "#185FA5", "#0F6E56", "#BA7517", "#9C2B22", "#666666", "#E6F1FB"

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--", "figure.dpi": 180,
})


# ── Figure 1 : convergence de l'autoencodeur ────────────────────
def fig_autoencoder_convergence():
    with open(f"{INTERIM_DIR}/autoencoder_history.json") as f:
        history = json.load(f)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history["loss"], color=BLUE, lw=2, label="Train")
    ax.plot(history["val_loss"], color=AMBER, lw=2, ls="--", label="Validation")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Reconstruction MSE")
    ax.set_title("LSTM Autoencoder Convergence", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig1_autoencoder_convergence.png", bbox_inches="tight")
    plt.close()
    print("  [OK] fig1_autoencoder_convergence.png")


# ── Figure 2 : performance d'ablation (9 combinaisons) ──────────
def fig_ablation_performance():
    df = pd.read_csv(f"{INTERIM_DIR}/ablation_results.csv").sort_values("AUC-ROC")
    colors = {"Aggregated": GRAY, "Embedding": AMBER, "Combined": BLUE}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    labels = df["Config"] + " / " + df["Model"]
    bars = ax.barh(labels, df["AUC-ROC"], xerr=df["AUC_std"],
                    color=[colors[c] for c in df["Config"]], capsize=3)
    for bar, val in zip(bars, df["AUC-ROC"]):
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", fontsize=9)
    ax.set_xlabel("AUC-ROC (mean +/- std across folds)")
    ax.set_xlim(df["AUC-ROC"].min() - 0.03, df["AUC-ROC"].max() + 0.03)
    ax.set_title("Performance by Feature Configuration and Model", fontweight="bold")
    handles = [plt.Rectangle((0,0),1,1, color=c) for c in colors.values()]
    ax.legend(handles, colors.keys(), loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig2_ablation_performance.png", bbox_inches="tight")
    plt.close()
    print("  [OK] fig2_ablation_performance.png")


# ── Figure 3 : baselines vs notre meilleure configuration ────────
def fig_baselines_vs_ours():
    base = pd.read_csv(f"{INTERIM_DIR}/baselines_results.csv", index_col=0)
    abl = pd.read_csv(f"{INTERIM_DIR}/ablation_results.csv").sort_values("AUC-ROC", ascending=False)
    best = abl.iloc[0]

    names = list(base.index) + [f"Our Pipeline\n({best['Config']}/{best['Model']})"]
    aucs = list(base["auc_mean"]) + [best["AUC-ROC"]]
    errs = list(base["auc_std"]) + [best["AUC_std"]]
    colors = [GRAY, GRAY, RED, GREEN]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, aucs, yerr=errs, color=colors[:len(names)], capsize=4, alpha=0.9)
    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.01, f"{val:.3f}", ha="center", fontweight="bold")
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0.80, 1.0)
    ax.set_title("Literature Baselines vs Proposed Pipeline", fontweight="bold")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig3_baselines_vs_ours.png", bbox_inches="tight")
    plt.close()
    print("  [OK] fig3_baselines_vs_ours.png")


# ── Figure 4 : importance SHAP (bar chart) ───────────────────────
def fig_shap_bar():
    top = pd.read_csv(f"{INTERIM_DIR}/shap_top_features.csv").sort_values("mean_abs_shap")
    colors = [AMBER if f.startswith("lstm_") else BLUE for f in top["feature"]]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.barh(top["feature"], top["mean_abs_shap"], color=colors)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Importance — Best Model", fontweight="bold")
    handles = [plt.Rectangle((0,0),1,1, color=BLUE), plt.Rectangle((0,0),1,1, color=AMBER)]
    ax.legend(handles, ["Aggregated descriptor", "LSTM embedding dimension"], loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig4_shap_top_features.png", bbox_inches="tight")
    plt.close()
    print("  [OK] fig4_shap_top_features.png")


# ── Figure 5 : SHAP summary plot (essaim de points) ──────────────
def fig_shap_beeswarm():
    sv = np.load(f"{INTERIM_DIR}/shap_values.npy")
    X_sample = np.load(f"{INTERIM_DIR}/shap_X_sample.npy")
    with open(f"{INTERIM_DIR}/shap_feature_names.txt") as f:
        feature_names = f.read().splitlines()

    plt.figure(figsize=(8, 6))
    shap.summary_plot(sv, X_sample, feature_names=feature_names, show=False, max_display=15)
    plt.title("SHAP Summary Plot", fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig5_shap_summary_beeswarm.png", bbox_inches="tight", dpi=180)
    plt.close()
    print("  [OK] fig5_shap_summary_beeswarm.png")


# ── Figure 6 : matrice de confusion (out-of-fold, meilleure config) ─
def fig_confusion_matrix():
    cohort = pd.read_parquet(f"{INTERIM_DIR}/cohort.parquet")
    feats = pd.read_parquet(f"{INTERIM_DIR}/aggregated_features.parquet")
    embeddings = np.load(f"{INTERIM_DIR}/embeddings.npy")
    abl = pd.read_csv(f"{INTERIM_DIR}/ablation_results.csv").sort_values("AUC-ROC", ascending=False)
    best = abl.iloc[0]

    X_agg = feats[AGGREGATED_FEATURE_NAMES].values
    if best["Config"] == "Aggregated":
        X = X_agg
    elif best["Config"] == "Embedding":
        X = embeddings
    else:
        X = np.concatenate([X_agg, embeddings], axis=1)

    y = cohort["dropout"].values
    groups = cohort["id_student"].values

    if best["Model"] == "XGBoost":
        import xgboost as xgb
        clf = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                 eval_metric="logloss", random_state=RANDOM_SEED)
    elif best["Model"] == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced",
                                       random_state=RANDOM_SEED, n_jobs=-1)
    else:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        clf = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])

    gkf = GroupKFold(n_splits=5)
    y_pred = cross_val_predict(clf, X, y, groups=groups, cv=gkf, method="predict")

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(5.8, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    classes = ["No dropout", "Dropout"]
    ax.set_xticks([0,1]); ax.set_xticklabels(classes)
    ax.set_yticks([0,1]); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=13, fontweight="bold",
                    color="white" if cm[i,j] > cm.max()/2 else "black")
    ax.set_title(f"Confusion Matrix — {best['Config']}/{best['Model']}\n(out-of-fold predictions)",
                 fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig6_confusion_matrix.png", bbox_inches="tight")
    plt.close()
    print("  [OK] fig6_confusion_matrix.png")


# ── Figure 7 : erreur de reconstruction par classe (verif. qualitative) ─
def fig_reconstruction_by_class():
    cohort = pd.read_parquet(f"{INTERIM_DIR}/cohort.parquet")
    recon_mse = np.load(f"{INTERIM_DIR}/recon_mse.npy")
    df = pd.DataFrame({"dropout": cohort["dropout"].map({0: "No dropout", 1: "Dropout"}),
                        "recon_mse": recon_mse})
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    df.boxplot(column="recon_mse", by="dropout", ax=ax, showfliers=False,
               patch_artist=True, boxprops=dict(facecolor=LIGHT))
    ax.set_title("Reconstruction Error by Class", fontweight="bold")
    ax.set_ylabel("Reconstruction MSE"); ax.set_xlabel("")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig7_reconstruction_by_class.png", bbox_inches="tight")
    plt.close()
    print("  [OK] fig7_reconstruction_by_class.png")


# ── Table 1 : tableau d'ablation (image) ─────────────────────────
def table_ablation():
    df = pd.read_csv(f"{INTERIM_DIR}/ablation_results.csv")
    fig, ax = plt.subplots(figsize=(9, 0.5*len(df)+1.5))
    ax.axis("off")
    tbl = ax.table(cellText=df.round(4).values, colLabels=df.columns, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1.0, 1.6)
    for j in range(len(df.columns)):
        tbl[(0,j)].set_facecolor(BLUE)
        tbl[(0,j)].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(df)+1):
        for j in range(len(df.columns)):
            tbl[(i,j)].set_facecolor(LIGHT if i % 2 == 0 else "white")
    ax.set_title("Table 1 — Complete Ablation Results", fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/table1_ablation.png", bbox_inches="tight")
    plt.close()
    print("  [OK] table1_ablation.png")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    print("Generation des figures...")
    fig_autoencoder_convergence()
    fig_ablation_performance()
    fig_baselines_vs_ours()
    fig_shap_bar()
    fig_shap_beeswarm()
    fig_confusion_matrix()
    fig_reconstruction_by_class()
    table_ablation()
    print(f"\n[OK] Toutes les figures sont dans {FIG_DIR}/")
