"""
features_aggregated.py
─────────────────────────────────────────────────────────────────
Section 4.5 de la methodologie — Descripteurs agreges du clickstream.

POINT METHODOLOGIQUE IMPORTANT (corrige par rapport a la V13) :
Cette fonction est l'UNIQUE definition des 8 descripteurs agreges.
baselines.py, lstm_autoencoder.py (config "Combined") et evaluation.py
importent tous AGGREGATED_FEATURE_NAMES et compute_aggregated_features
d'ICI — aucun module ne doit recalculer sa propre variante. C'est cette
divergence silencieuse (B1 utilisant plus de features que ce que le
texte de la V13 annoncait) qui avait fausse la comparaison aux baselines.

Usage :
    python3 features_aggregated.py
─────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import pandas as pd

INTERIM_DIR = os.path.join(os.path.dirname(__file__), "..", "interim")

ASSESSMENT_ACTIVITY_TYPES = {"quiz", "externalquiz", "questionnaire"}

AGGREGATED_FEATURE_NAMES = [
    "total_clicks",        # volume total d'interaction
    "active_weeks",        # nombre de semaines (0..38) avec au moins un clic
    "active_days",         # nombre de jours distincts avec au moins un clic
    "resource_diversity",  # nombre de types d'activite distincts utilises
    "assessment_clicks",   # clics sur des activites liees a l'evaluation
    "peak_week_clicks",    # clics de la semaine la plus active
    "avg_weekly_clicks",   # moyenne des clics par semaine active
    "click_std",           # ecart-type des clics hebdomadaires (regularite)
]


def compute_aggregated_features(student_vle: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les 8 descripteurs agreges pour chaque etudiant de la cohorte,
    a partir du clickstream BRUT (pas du tenseur deja agrege de preprocessing.py —
    volontairement independant, pour servir de point de comparaison a la
    representation LSTM).
    """
    keys = ["code_module", "code_presentation", "id_student"]

    sv = student_vle.merge(cohort[keys], on=keys, how="inner")
    sv["week"] = (sv["date"] // 7).clip(lower=0, upper=38)

    # --- Volume et diversite ---
    g = sv.groupby(keys, observed=True)
    total_clicks = g["sum_click"].sum().rename("total_clicks")
    active_days = g["date"].nunique().rename("active_days")
    active_weeks = g["week"].nunique().rename("active_weeks")
    resource_diversity = sv.groupby(keys, observed=True)["id_site"].nunique().rename("resource_diversity")

    # --- Clics lies a l'evaluation (necessite le type d'activite) ---
    vle = pd.read_parquet(f"{INTERIM_DIR}/vle.parquet")[["id_site", "activity_type"]]
    sv_typed = sv.merge(vle, on="id_site", how="left")
    assessment_mask = sv_typed["activity_type"].isin(ASSESSMENT_ACTIVITY_TYPES)
    assessment_clicks = (
        sv_typed[assessment_mask].groupby(keys, observed=True)["sum_click"].sum()
        .rename("assessment_clicks")
    )

    # --- Statistiques hebdomadaires (peak / moyenne / ecart-type) ---
    weekly = sv.groupby(keys + ["week"], observed=True)["sum_click"].sum().reset_index()
    weekly_stats = weekly.groupby(keys, observed=True)["sum_click"].agg(
        peak_week_clicks="max", avg_weekly_clicks="mean", click_std="std"
    )

    feats = pd.concat(
        [total_clicks, active_weeks, active_days, resource_diversity, assessment_clicks, weekly_stats],
        axis=1,
    )
    feats = feats.reindex(columns=AGGREGATED_FEATURE_NAMES)
    feats[["total_clicks", "active_weeks", "active_days", "resource_diversity", "assessment_clicks"]] = (
        feats[["total_clicks", "active_weeks", "active_days", "resource_diversity", "assessment_clicks"]].fillna(0)
    )
    feats["click_std"] = feats["click_std"].fillna(0)  # etudiants actifs une seule semaine -> std indefini -> 0
    feats = feats.reset_index()

    # Reindexer explicitement sur TOUTE la cohorte (etudiants sans ligne dans sv = déjà exclus en amont,
    # donc ceci est surtout une sécurité de cohérence d'ordre)
    feats = cohort[keys].merge(feats, on=keys, how="left")
    assert feats[AGGREGATED_FEATURE_NAMES].isna().sum().sum() == 0, "NaN residuels dans les features agregees !"

    return feats


def validation_report(feats: pd.DataFrame, cohort: pd.DataFrame):
    print("=" * 70)
    print("RAPPORT DE VALIDATION — features_aggregated.py")
    print("=" * 70)
    print(f"\nForme : {feats.shape}  (attendu : {len(cohort)} x {3 + len(AGGREGATED_FEATURE_NAMES)})")
    assert len(feats) == len(cohort)

    print("\nStatistiques descriptives :")
    print(feats[AGGREGATED_FEATURE_NAMES].describe().T.to_string(float_format=lambda x: f"{x:10.2f}"))

    print("\nCorrelation avec la cible (dropout) — signal univarie attendu :")
    merged = feats.merge(cohort[["code_module", "code_presentation", "id_student", "dropout"]],
                          on=["code_module", "code_presentation", "id_student"])
    corr = merged[AGGREGATED_FEATURE_NAMES + ["dropout"]].corr()["dropout"].drop("dropout")
    print(corr.sort_values().to_string(float_format=lambda x: f"{x:+.3f}"))
    print("\n(toutes negatives est attendu : moins d'activite -> plus de risque de decrochage)")
    print("=" * 70)


if __name__ == "__main__":
    cohort = pd.read_parquet(f"{INTERIM_DIR}/cohort.parquet")
    student_vle = pd.read_parquet(f"{INTERIM_DIR}/student_vle.parquet")

    feats = compute_aggregated_features(student_vle, cohort)
    validation_report(feats, cohort)

    feats.to_parquet(f"{INTERIM_DIR}/aggregated_features.parquet")
    print(f"\n[OK] Features agregees sauvegardees -> {INTERIM_DIR}/aggregated_features.parquet")
