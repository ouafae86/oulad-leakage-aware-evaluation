"""
baselines.py
─────────────────────────────────────────────────────────────────
Section 4.4 de la methodologie — Reproduction des baselines de la litterature.

  B1 : Random Forest sur les 8 descripteurs agreges (features_aggregated.py)
  B2 : XGBoost sur descripteurs agreges + variables demographiques
  B3 : LSTM supervise (meme architecture d'entree que lstm_autoencoder.py,
       mais entraine directement sur le label — sert de reference haute
       pour situer l'apport du LSTM NON supervise)

POINT METHODOLOGIQUE (corrige par rapport a la V13) :
- B1 utilise EXACTEMENT les 8 features de AGGREGATED_FEATURE_NAMES
  (importees depuis features_aggregated.py) — aucun sous-ensemble
  cache, aucune divergence entre le texte et le code.
- Toutes les baselines sont evaluees avec le MEME protocole de
  validation croisee (GroupKFold sur id_student) que notre pipeline,
  pour une comparaison honnete. GroupKFold est necessaire car 3808
  etudiants apparaissent dans plusieurs (module, presentation) —
  un simple StratifiedKFold les laisserait fuiter entre train et test.

Usage :
    python3 baselines.py [--quick]
─────────────────────────────────────────────────────────────────
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from features_aggregated import AGGREGATED_FEATURE_NAMES

INTERIM_DIR = os.path.join(os.path.dirname(__file__), "..", "interim")
RANDOM_SEED = 42
DEMOGRAPHIC_COLS = ["gender", "region", "highest_education", "imd_band",
                    "age_band", "disability", "num_of_prev_attempts", "studied_credits"]


def _cv_evaluate(X, y, groups, build_model_fn, n_splits=5):
    """Evaluation par validation croisee groupee (GroupKFold sur id_student)."""
    gkf = GroupKFold(n_splits=n_splits)
    aucs, accs, f1s = [], [], []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        model = build_model_fn()
        X_tr = X.iloc[tr] if hasattr(X, "iloc") else X[tr]
        X_te = X.iloc[te] if hasattr(X, "iloc") else X[te]
        model.fit(X_tr, y[tr])
        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        aucs.append(roc_auc_score(y[te], proba))
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred))
    return {"auc_mean": np.mean(aucs), "auc_std": np.std(aucs),
            "acc_mean": np.mean(accs), "f1_mean": np.mean(f1s), "auc_folds": aucs}


def run_b1_random_forest(feats: pd.DataFrame, cohort: pd.DataFrame, n_splits=5) -> dict:
    """B1 : Random Forest sur les 8 descripteurs agreges uniquement."""
    X = feats[AGGREGATED_FEATURE_NAMES]
    y = cohort["dropout"].values
    groups = cohort["id_student"].values

    build = lambda: RandomForestClassifier(n_estimators=200, max_depth=12,
                                            class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)
    return _cv_evaluate(X, y, groups, build, n_splits)


def run_b2_xgboost_demo(feats: pd.DataFrame, cohort: pd.DataFrame, n_splits=5) -> dict:
    """B2 : XGBoost sur descripteurs agreges + variables demographiques."""
    import xgboost as xgb

    demo = cohort[DEMOGRAPHIC_COLS].copy()
    X = pd.concat([feats[AGGREGATED_FEATURE_NAMES].reset_index(drop=True),
                   demo.reset_index(drop=True)], axis=1)
    y = cohort["dropout"].values
    groups = cohort["id_student"].values

    cat_cols = [c for c in DEMOGRAPHIC_COLS if demo[c].dtype.name in ("category", "object")]
    num_cols = [c for c in X.columns if c not in cat_cols]

    def build():
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ], remainder="passthrough")
        clf = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                 eval_metric="logloss", random_state=RANDOM_SEED,
                                 scale_pos_weight=(y == 0).sum() / (y == 1).sum())
        return Pipeline([("pre", pre), ("clf", clf)])

    return _cv_evaluate(X, y, groups, build, n_splits)


def run_b3_supervised_lstm(sequences: np.ndarray, cohort: pd.DataFrame,
                            n_splits=5, epochs=15) -> dict:
    """B3 : LSTM supervise (meme entree que l'autoencodeur, entraine avec les labels)."""
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    from lstm_autoencoder import normalize_sequences

    y = cohort["dropout"].values
    groups = cohort["id_student"].values
    normalized, _, _ = normalize_sequences(sequences)

    gkf = GroupKFold(n_splits=n_splits)
    aucs, accs, f1s = [], [], []
    n_timesteps, n_features = sequences.shape[1], sequences.shape[2]

    for tr, te in gkf.split(normalized, y, groups):
        tf.random.set_seed(RANDOM_SEED)
        inputs = layers.Input(shape=(n_timesteps, n_features))
        x = layers.Masking()(inputs)
        x = layers.LSTM(64)(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        model = models.Model(inputs, out)
        model.compile(optimizer="adam", loss="binary_crossentropy")
        model.fit(normalized[tr], y[tr], epochs=epochs, batch_size=128, verbose=0,
                  validation_split=0.1,
                  callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)])
        proba = model.predict(normalized[te], verbose=0).ravel()
        pred = (proba >= 0.5).astype(int)
        aucs.append(roc_auc_score(y[te], proba))
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred))

    return {"auc_mean": np.mean(aucs), "auc_std": np.std(aucs),
            "acc_mean": np.mean(accs), "f1_mean": np.mean(f1s), "auc_folds": aucs}


def validation_report(results: dict):
    print("=" * 70)
    print("RAPPORT — baselines.py")
    print("=" * 70)
    for name, r in results.items():
        print(f"\n{name} :")
        print(f"  AUC-ROC = {r['auc_mean']:.4f} (+/- {r['auc_std']:.4f})   "
              f"Acc = {r['acc_mean']:.4f}   F1 = {r['f1_mean']:.4f}")
        print(f"  AUC par fold : {[round(a,4) for a in r['auc_folds']]}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-lstm", action="store_true", help="Sauter B3 (LSTM supervise, le plus lent)")
    args = parser.parse_args()
    n_splits = 2 if args.quick else 5
    epochs = 3 if args.quick else 15

    cohort = pd.read_parquet(f"{INTERIM_DIR}/cohort.parquet")
    feats = pd.read_parquet(f"{INTERIM_DIR}/aggregated_features.parquet")

    results = {}
    print("[B1] Random Forest (descripteurs agreges)...")
    results["B1_RandomForest"] = run_b1_random_forest(feats, cohort, n_splits)

    print("[B2] XGBoost (descripteurs agreges + demographie)...")
    results["B2_XGBoost_demo"] = run_b2_xgboost_demo(feats, cohort, n_splits)

    if not args.skip_lstm:
        print("[B3] LSTM supervise (peut prendre plusieurs minutes)...")
        sequences = np.load(f"{INTERIM_DIR}/sequences.npy")
        results["B3_LSTM_supervised"] = run_b3_supervised_lstm(sequences, cohort, n_splits, epochs)

    validation_report(results)

    pd.DataFrame(results).T.to_csv(f"{INTERIM_DIR}/baselines_results.csv")
    print(f"\n[OK] Resultats sauvegardes -> {INTERIM_DIR}/baselines_results.csv")
    if args.quick:
        print("[ATTENTION] --quick actif (2 folds, peu d'epochs) : validation du code uniquement.")
