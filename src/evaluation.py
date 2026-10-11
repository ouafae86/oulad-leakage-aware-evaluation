"""
evaluation.py
─────────────────────────────────────────────────────────────────
Section 4.7 de la methodologie — Ablation, evaluation et tests statistiques.

Compare 3 configurations de features :
  - Aggregated  : les 8 descripteurs agreges seuls
  - Embedding   : les embeddings LSTM non supervises seuls (latent_dim)
  - Combined    : les deux concatenes

... x 3 classifieurs (LR, RF, XGBoost), avec le MEME protocole
GroupKFold (par id_student) que baselines.py, pour une comparaison
strictement homogene.

Test statistique : Wilcoxon signed-rank appareille sur les AUC par
fold entre la meilleure config et B3 (LSTM supervise) — teste si la
difference est significative, PAS juste si Combined > Aggregated
(comparaison la plus interessante pour l'article : le non-supervise
rivalise-t-il avec le supervise ?).

Usage :
    python3 evaluation.py [--quick]
─────────────────────────────────────────────────────────────────
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from scipy.stats import t as t_dist
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from features_aggregated import AGGREGATED_FEATURE_NAMES

INTERIM_DIR = os.path.join(os.path.dirname(__file__), "..", "interim")
RANDOM_SEED = 42

CLASSIFIERS = {
    "LogisticRegression": lambda: Pipeline([("scale", StandardScaler()),
                                             ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                                                         random_state=RANDOM_SEED))]),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=200, max_depth=12,
                                                     class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1),
    "XGBoost": lambda: __import__("xgboost").XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                                             eval_metric="logloss", random_state=RANDOM_SEED),
}


def build_feature_configs(feats: pd.DataFrame, embeddings: np.ndarray):
    X_agg = feats[AGGREGATED_FEATURE_NAMES].values
    X_emb = embeddings
    X_comb = np.concatenate([X_agg, X_emb], axis=1)
    return {"Aggregated": X_agg, "Embedding": X_emb, "Combined": X_comb}


def run_ablation(feats, embeddings, cohort, n_splits=5):
    configs = build_feature_configs(feats, embeddings)
    y = cohort["dropout"].values
    groups = cohort["id_student"].values
    gkf = GroupKFold(n_splits=n_splits)

    rows = []
    fold_aucs = {}  # (config, model) -> liste d'AUC par fold, pour les tests stat
    for config_name, X in configs.items():
        for model_name, build_fn in CLASSIFIERS.items():
            aucs, accs, f1s = [], [], []
            for tr, te in gkf.split(X, y, groups):
                model = build_fn()
                model.fit(X[tr], y[tr])
                proba = model.predict_proba(X[te])[:, 1]
                pred = (proba >= 0.5).astype(int)
                aucs.append(roc_auc_score(y[te], proba))
                accs.append(accuracy_score(y[te], pred))
                f1s.append(f1_score(y[te], pred))
            fold_aucs[(config_name, model_name)] = aucs
            rows.append({
                "Config": config_name, "Model": model_name,
                "AUC-ROC": np.mean(aucs), "AUC_std": np.std(aucs),
                "Accuracy": np.mean(accs), "F1": np.mean(f1s),
            })
    results_df = pd.DataFrame(rows).sort_values("AUC-ROC", ascending=False).reset_index(drop=True)
    return results_df, fold_aucs


def tost_equivalence(a: np.ndarray, b: np.ndarray, margin: float):
    """
    Two One-Sided Tests (TOST) pour l'equivalence de deux series appariees.
    H0 (rejetee si equivalence) : |mean(a-b)| >= margin.
    Retourne le p le plus defavorable des deux tests unilateraux (test standard).
    """
    diff = np.asarray(a) - np.asarray(b)
    n = len(diff)
    mean_d, se = diff.mean(), diff.std(ddof=1) / np.sqrt(n)
    t1 = (mean_d - (-margin)) / se       # H0: diff <= -margin
    t2 = (margin - mean_d) / se          # H0: diff >= margin
    p1, p2 = 1 - t_dist.cdf(t1, df=n - 1), 1 - t_dist.cdf(t2, df=n - 1)
    ci90 = t_dist.interval(0.90, df=n - 1, loc=mean_d, scale=se)
    return {"mean_diff": mean_d, "ci90_low": ci90[0], "ci90_high": ci90[1],
            "p_tost": max(p1, p2), "n": n, "margin": margin}


def statistical_comparison(fold_aucs: dict, baselines_results: pd.DataFrame,
                            equivalence_margin: float = None):
    """
    Compare la meilleure configuration a B3 (LSTM supervise) de DEUX facons :
      1) Wilcoxon signed-rank (test de DIFFERENCE — un p non significatif ne
         prouve PAS l'equivalence, seulement l'absence de preuve de difference).
      2) TOST (test D'EQUIVALENCE proprement dit) — c'est celui-ci qui doit
         etayer une affirmation du type "le modele non supervise egale le
         modele supervise" dans le texte de l'article.
    La marge d'equivalence, si non fournie, est fixee par defaut au plus petit
    ecart observe entre deux baselines consecutives dans baselines_results
    (ecart considere comme la plus petite difference "methodologiquement
    significative" dans cette etude) — a documenter explicitement dans le texte.
    """
    best_key = max(fold_aucs, key=lambda k: np.mean(fold_aucs[k]))
    best_aucs = np.array(fold_aucs[best_key])
    print(f"\nMeilleure configuration : {best_key} -> AUC moyen = {np.mean(best_aucs):.4f}")

    if "B3_LSTM_supervised" not in baselines_results.index:
        print("[Info] B3 absent des resultats baselines (probablement --skip-lstm) : comparaison statistique ignoree.")
        return None

    b3_aucs = baselines_results.loc["B3_LSTM_supervised", "auc_folds"]
    b3_aucs = np.array(eval(b3_aucs) if isinstance(b3_aucs, str) else b3_aucs)

    if len(b3_aucs) != len(best_aucs):
        print(f"[Attention] Nombre de folds different (best={len(best_aucs)}, B3={len(b3_aucs)}) "
              "— comparaison appariee non valide, resultats non calcules.")
        return None

    diff = np.mean(best_aucs) - np.mean(b3_aucs)
    stat, p_wilcoxon = wilcoxon(best_aucs, b3_aucs)
    print(f"[Wilcoxon - test de difference] diff moyenne = {diff:+.4f}, p = {p_wilcoxon:.4f}  "
          "(non significatif != preuve d'equivalence, cf. TOST ci-dessous)")

    if equivalence_margin is None:
        base_aucs = baselines_results["auc_mean"].sort_values()
        equivalence_margin = float(base_aucs.diff().dropna().min())
        print(f"[Marge d'equivalence] non fournie -> deduite automatiquement du plus petit ecart "
              f"entre deux baselines consecutives = {equivalence_margin:.4f}")

    tost = tost_equivalence(best_aucs, b3_aucs, equivalence_margin)
    verdict = "EQUIVALENCE CONFIRMEE" if tost["p_tost"] < 0.05 else "equivalence NON confirmee"
    print(f"[TOST - test d'equivalence, marge=±{equivalence_margin:.4f}] "
          f"IC90%=[{tost['ci90_low']:+.4f}, {tost['ci90_high']:+.4f}]  p_TOST={tost['p_tost']:.4f}  -> {verdict}")

    return {"best_config": str(best_key), "diff_vs_b3": diff, "p_wilcoxon": p_wilcoxon,
            "equivalence_margin": equivalence_margin, **tost}


def validation_report(results_df):
    print("=" * 70)
    print("RAPPORT — evaluation.py (ablation)")
    print("=" * 70)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    n_splits = 2 if args.quick else 5

    cohort = pd.read_parquet(f"{INTERIM_DIR}/cohort.parquet")
    feats = pd.read_parquet(f"{INTERIM_DIR}/aggregated_features.parquet")
    embeddings = np.load(f"{INTERIM_DIR}/embeddings.npy")

    results_df, fold_aucs = run_ablation(feats, embeddings, cohort, n_splits)
    validation_report(results_df)

    try:
        baselines_results = pd.read_csv(f"{INTERIM_DIR}/baselines_results.csv", index_col=0)
        stat_result = statistical_comparison(fold_aucs, baselines_results)
    except FileNotFoundError:
        print("[Info] baselines_results.csv absent — executez baselines.py avant pour la comparaison statistique.")
        stat_result = None

    results_df.to_csv(f"{INTERIM_DIR}/ablation_results.csv", index=False)
    print(f"\n[OK] Resultats d'ablation sauvegardes -> {INTERIM_DIR}/ablation_results.csv")

    # AUC par fold pour CHAQUE configuration (essentiel pour tout test statistique
    # ulterieur, y compris hors de ce script — sans ca, le TOST/Wilcoxon ne sont
    # pas reproductibles a partir des seuls fichiers interim/).
    import json
    fold_aucs_serializable = {f"{k[0]}__{k[1]}": v for k, v in fold_aucs.items()}
    with open(f"{INTERIM_DIR}/ablation_fold_aucs.json", "w") as f:
        json.dump(fold_aucs_serializable, f, indent=2)
    print(f"[OK] AUC par fold (toutes configs) sauvegardees -> {INTERIM_DIR}/ablation_fold_aucs.json")

    if stat_result is not None:
        with open(f"{INTERIM_DIR}/statistical_comparison.json", "w") as f:
            json.dump(stat_result, f, indent=2, default=float)
        print(f"[OK] Comparaison statistique (Wilcoxon + TOST) sauvegardee -> {INTERIM_DIR}/statistical_comparison.json")
    if args.quick:
        print("[ATTENTION] --quick actif (2 folds) : validation du code uniquement.")
