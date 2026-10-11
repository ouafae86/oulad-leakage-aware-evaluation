"""
preprocessing.py
─────────────────────────────────────────────────────────────────
Section 4.3 de la methodologie — Definition de la cohorte d'analyse
et construction des sequences hebdomadaires de clics.

Deux responsabilites strictes :
  1) build_analysis_cohort  : applique le critere d'exclusion decide
     avec l'utilisateur (etudiants sans aucun clic enregistre).
  2) build_weekly_sequences : transforme le clickstream brut
     (10.6M lignes) en un tenseur dense (n_etudiants, n_semaines,
     n_types_activite) consommable par le LSTM.

Usage :
    python3 preprocessing.py
─────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import pandas as pd

INTERIM_DIR = os.path.join(os.path.dirname(__file__), "..", "interim")
MAX_WEEKS = 39  # semaines 0..38 (couvre [-25, 269] jours apres clipping des dates negatives)


# ══════════════════════════════════════════════════════════════
# 1) COHORTE D'ANALYSE — exclusion des etudiants sans clic
# ══════════════════════════════════════════════════════════════

def build_analysis_cohort(student_table: pd.DataFrame, student_vle: pd.DataFrame) -> pd.DataFrame:
    """
    Exclut les etudiants sans aucun evenement dans studentVle.

    Justification (fixee avec l'utilisateur le [date]) : ces etudiants
    (3365 dans le jeu complet, 10.3%) se desinscrivent en tres grande
    majorite (89%) avant meme le debut du cours (mediane de
    date_unregistration = -18 jours) — il n'y a donc aucune sequence
    comportementale a construire pour eux, et les garder introduirait
    un raccourci trivial ("pas de clic = decrochage") plutot qu'un
    vrai signal comportemental appris.
    """
    keys = ["code_module", "code_presentation", "id_student"]
    active_keys = set(map(tuple, student_vle[keys].drop_duplicates().values))

    mask = student_table[keys].apply(tuple, axis=1).isin(active_keys)
    cohort = student_table[mask].reset_index(drop=True)

    n_excluded = len(student_table) - len(cohort)
    print(f"[Cohorte] {len(student_table)} etudiants -> {len(cohort)} apres exclusion "
          f"({n_excluded} exclus, {100*n_excluded/len(student_table):.1f}%)")
    return cohort


# ══════════════════════════════════════════════════════════════
# 2) SEQUENCES HEBDOMADAIRES — tenseur (n_etudiants, semaines, activites)
# ══════════════════════════════════════════════════════════════

def build_weekly_sequences(student_vle: pd.DataFrame, vle: pd.DataFrame,
                            cohort: pd.DataFrame, max_weeks: int = MAX_WEEKS):
    """
    Construit un tenseur dense de clics hebdomadaires par type d'activite.

    Regles de construction (a documenter dans la methodologie) :
      - semaine = date // 7 (date = jour relatif au debut du module, jour 0)
      - les clics a date < 0 (acces pre-cours, 6.47% des evenements) sont
        replies sur la semaine 0 plutot que rejetes : c'est un signal
        d'engagement precoce legitime (preparation avant le cours),
        pas du bruit.
      - les semaines > max_weeks-1 sont repliees sur la derniere semaine
        (aucune occurrence dans ce jeu de donnees, filet de securite).
      - seuls les (module, presentation, id_student) presents dans la
        cohorte filtree sont conserves (coherence avec build_analysis_cohort).

    Retourne :
      sequences      : np.ndarray shape (n_etudiants, max_weeks, n_activites), dtype float32
      activity_types : liste des types d'activite (ordre = axe 3 du tenseur)
      student_order   : liste des triplets (module, presentation, id_student),
                        dans l'ordre exact des lignes de `sequences`
    """
    keys = ["code_module", "code_presentation", "id_student"]

    # Index etudiant (ordre = cohort, deterministe) — jointure vectorisee, pas d'apply()
    student_order = list(map(tuple, cohort[keys].values))
    cohort_idx = cohort[keys].copy()
    cohort_idx["student_idx"] = np.arange(len(cohort), dtype="int32")

    # Restreindre le clickstream a la cohorte ET recuperer student_idx en une seule jointure
    sv = student_vle.merge(cohort_idx, on=keys, how="inner")

    # Rattacher le type d'activite (vle.csv) a chaque evenement de clic
    sv = sv.merge(vle[["id_site", "activity_type"]], on="id_site", how="left")
    n_missing_type = sv["activity_type"].isna().sum()
    if n_missing_type:
        print(f"[Attention] {n_missing_type} evenements sans activity_type retrouve (id_site absent de vle.csv) — ignores.")
        sv = sv.dropna(subset=["activity_type"])

    activity_types = sorted(vle["activity_type"].unique().tolist())
    act_to_idx = {a: i for i, a in enumerate(activity_types)}

    # Semaine : repli des dates negatives sur 0, repli du haut sur max_weeks-1
    sv["week"] = (sv["date"] // 7).clip(lower=0, upper=max_weeks - 1).astype("int16")

    # Agregation : somme des clics par (etudiant, semaine, type d'activite)
    agg = (
        sv.groupby(["student_idx", "week", "activity_type"], observed=True)["sum_click"]
        .sum()
        .reset_index()
    )

    n_students = len(student_order)
    n_activities = len(activity_types)
    sequences = np.zeros((n_students, max_weeks, n_activities), dtype="float32")

    act_idx = agg["activity_type"].map(act_to_idx).values
    sequences[agg["student_idx"].values, agg["week"].values, act_idx] = agg["sum_click"].values

    return sequences, activity_types, student_order


# ══════════════════════════════════════════════════════════════
# RAPPORT DE VALIDATION
# ══════════════════════════════════════════════════════════════

def validation_report(cohort, sequences, activity_types, student_order):
    print("\n" + "=" * 70)
    print("RAPPORT DE VALIDATION — preprocessing.py")
    print("=" * 70)

    print(f"\n--- Tenseur de sequences ---")
    print(f"  Forme                : {sequences.shape}  (etudiants, semaines, types d'activite)")
    print(f"  Types d'activite ({len(activity_types)}) : {activity_types}")
    print(f"  Taille memoire        : {sequences.nbytes/1e6:.1f} Mo")
    print(f"  Taux de cellules non nulles (sparsite) : {100*(sequences>0).mean():.2f}%")

    print(f"\n--- Coherence avec la cohorte ---")
    assert sequences.shape[0] == len(cohort), "Desalignement etudiants/sequences !"
    assert len(student_order) == len(cohort)
    print(f"  {sequences.shape[0]} etudiants dans le tenseur == {len(cohort)} dans la cohorte  OK")

    print(f"\n--- Sanity check : total clics reconstruit vs somme brute (echantillon) ---")
    total_reconstructed = sequences.sum()
    print(f"  Somme totale des clics dans le tenseur : {total_reconstructed:,.0f}")

    print(f"\n--- Exemple : profil d'un etudiant (le premier de la cohorte) ---")
    ex = sequences[0]
    weekly_total = ex.sum(axis=1)
    print(f"  Etudiant {student_order[0]}")
    print(f"  Clics totaux (12 premieres semaines) : {weekly_total[:12].round(0).tolist()}")

    print("\n" + "=" * 70)
    print("VALIDATION TERMINEE")
    print("=" * 70)


if __name__ == "__main__":
    student_table = pd.read_parquet(f"{INTERIM_DIR}/student_table.parquet")
    student_vle = pd.read_parquet(f"{INTERIM_DIR}/student_vle.parquet")
    vle = pd.read_parquet(f"{INTERIM_DIR}/vle.parquet")

    cohort = build_analysis_cohort(student_table, student_vle)
    sequences, activity_types, student_order = build_weekly_sequences(student_vle, vle, cohort)
    validation_report(cohort, sequences, activity_types, student_order)

    # Sauvegarde pour les modules suivants
    cohort.to_parquet(f"{INTERIM_DIR}/cohort.parquet")
    np.save(f"{INTERIM_DIR}/sequences.npy", sequences)
    pd.DataFrame(student_order, columns=["code_module", "code_presentation", "id_student"]).to_parquet(
        f"{INTERIM_DIR}/student_order.parquet"
    )
    with open(f"{INTERIM_DIR}/activity_types.txt", "w") as f:
        f.write("\n".join(activity_types))
    print(f"\n[OK] Cohorte, sequences et metadonnees sauvegardees dans {INTERIM_DIR}/")
