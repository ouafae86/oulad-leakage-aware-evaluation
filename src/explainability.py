"""
explainability.py
─────────────────────────────────────────────────────────────────
Section 4.9 de la methodologie — Analyse SHAP du meilleur modele.

Explique les predictions du meilleur couple (Config, Modele) trouve
par evaluation.py, sur un echantillon (les explications SHAP exactes
sur l'ensemble complet seraient trop couteuses en TreeExplainer pour
XGBoost/RF au-dela de quelques centaines de points).

Usage :
    python3 explainability.py
─────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import pandas as pd
import shap

from features_aggregated import AGGREGATED_FEATURE_NAMES

INTERIM_DIR = os.path.join(os.path.dirname(__file__), "..", "interim")
RANDOM_SEED = 42


def _extract_class1_shap(shap_values) -> np.ndarray:
    """Gere les differents formats de sortie SHAP selon la version/le modele."""
    sv = np.array(shap_values)
    if isinstance(shap_values, list):
        return np.array(shap_values[1])
    if sv.ndim == 3:
        return sv[:, :, 1]
    return sv


def compute_shap_values(clf, X: np.ndarray, sample_size: int = 300):
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
    X_sample = X[idx]

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample)
    sv = _extract_class1_shap(shap_values)
    return sv, X_sample, explainer, idx


def get_top_features(sv: np.ndarray, feature_names: list, top_n: int = 15):
    mean_abs = np.abs(sv).mean(axis=0)
    ranking = np.argsort(mean_abs)[::-1][:top_n]
    return [(feature_names[i], float(mean_abs[i])) for i in ranking]


def train_best_model(feats, embeddings, cohort, config_name: str, model_name: str):
    """Reentraine le meilleur modele identifie par evaluation.py sur TOUTES les donnees
    (pour l'explication — plus de raison de garder un split train/test a ce stade)."""
    from sklearn.ensemble import RandomForestClassifier

    X_agg = feats[AGGREGATED_FEATURE_NAMES].values
    if config_name == "Aggregated":
        X, names = X_agg, AGGREGATED_FEATURE_NAMES
    elif config_name == "Embedding":
        emb_names = [f"lstm_{i:03d}" for i in range(embeddings.shape[1])]
        X, names = embeddings, emb_names
    else:  # Combined
        emb_names = [f"lstm_{i:03d}" for i in range(embeddings.shape[1])]
        X, names = np.concatenate([X_agg, embeddings], axis=1), AGGREGATED_FEATURE_NAMES + emb_names

    y = cohort["dropout"].values
    if model_name == "XGBoost":
        import xgboost as xgb
        clf = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                 eval_metric="logloss", random_state=RANDOM_SEED)
    else:
        clf = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced",
                                       random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X, y)
    return clf, X, names


def validation_report(sv, top_features):
    print("=" * 70)
    print("RAPPORT — explainability.py")
    print("=" * 70)
    print(f"\nForme des valeurs SHAP : {sv.shape}")
    print(f"\nTop 15 features par |SHAP| moyen :")
    for feat, val in top_features:
        print(f"  {feat:25s} {val:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    cohort = pd.read_parquet(f"{INTERIM_DIR}/cohort.parquet")
    feats = pd.read_parquet(f"{INTERIM_DIR}/aggregated_features.parquet")
    embeddings = np.load(f"{INTERIM_DIR}/embeddings.npy")

    try:
        ablation = pd.read_csv(f"{INTERIM_DIR}/ablation_results.csv")
        best_row = ablation.iloc[0]  # deja triee par AUC-ROC descendant dans evaluation.py
        config_name, model_name = best_row["Config"], best_row["Model"]
        if model_name not in ("XGBoost", "RandomForest"):
            print(f"[Info] Meilleur modele = {model_name} (non arborescent) -> "
                  "TreeExplainer non applicable, on utilise RandomForest+Combined par defaut pour l'illustration.")
            config_name, model_name = "Combined", "RandomForest"
    except FileNotFoundError:
        print("[Info] ablation_results.csv absent — utilisation par defaut Combined+RandomForest.")
        config_name, model_name = "Combined", "RandomForest"

    print(f"Modele explique : {model_name} / {config_name}")
    clf, X, feature_names = train_best_model(feats, embeddings, cohort, config_name, model_name)

    sv, X_sample, explainer, idx = compute_shap_values(clf, X, sample_size=300)
    top_features = get_top_features(sv, feature_names, top_n=15)
    validation_report(sv, top_features)

    np.save(f"{INTERIM_DIR}/shap_values.npy", sv)
    np.save(f"{INTERIM_DIR}/shap_X_sample.npy", X_sample)
    with open(f"{INTERIM_DIR}/shap_feature_names.txt", "w") as f:
        f.write("\n".join(feature_names))
    pd.DataFrame(top_features, columns=["feature", "mean_abs_shap"]).to_csv(
        f"{INTERIM_DIR}/shap_top_features.csv", index=False
    )
    print(f"\n[OK] Valeurs SHAP sauvegardees -> {INTERIM_DIR}/shap_values.npy")
