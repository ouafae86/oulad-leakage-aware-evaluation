"""
data_loading.py
─────────────────────────────────────────────────────────────────
Section 4.2 de la méthodologie — Chargement et validation des données OULAD.

Rôle STRICT de ce module : charger les 7 tables brutes, les typer
correctement, construire la cible binaire (décrochage), et produire
un rapport de validation. AUCUN feature engineering ici — ça, c'est
le rôle de preprocessing.py et features_aggregated.py.

Usage :
    python3 data_loading.py
─────────────────────────────────────────────────────────────────
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ══════════════════════════════════════════════════════════════
# CHARGEMENT DES TABLES BRUTES (dtypes optimisés dès le départ)
# ══════════════════════════════════════════════════════════════

def load_raw_tables(data_dir: str = DATA_DIR) -> dict:
    """Charge les 7 CSV OULAD avec des dtypes explicites et économes en mémoire."""

    cat_cols = {"code_module": "category", "code_presentation": "category"}

    courses = pd.read_csv(
        f"{data_dir}/courses.csv", dtype={**cat_cols, "module_presentation_length": "int16"}
    )

    student_info = pd.read_csv(
        f"{data_dir}/studentInfo.csv",
        dtype={
            **cat_cols, "id_student": "int32", "gender": "category",
            "region": "category", "highest_education": "category",
            "imd_band": "category", "age_band": "category",
            "num_of_prev_attempts": "int8", "studied_credits": "int16",
            "disability": "category", "final_result": "category",
        },
    )

    student_registration = pd.read_csv(
        f"{data_dir}/studentRegistration.csv",
        dtype={**cat_cols, "id_student": "int32"},
        # date_registration / date_unregistration : entiers signes avec NaN possibles -> float
    )
    student_registration["date_registration"] = pd.to_numeric(
        student_registration["date_registration"], errors="coerce"
    ).astype("Int32")
    student_registration["date_unregistration"] = pd.to_numeric(
        student_registration["date_unregistration"], errors="coerce"
    ).astype("Int32")

    assessments = pd.read_csv(
        f"{data_dir}/assessments.csv",
        dtype={**cat_cols, "id_assessment": "int32", "assessment_type": "category"},
    )
    assessments["date"] = pd.to_numeric(assessments["date"], errors="coerce").astype("Int32")
    assessments["weight"] = assessments["weight"].astype("float32")

    student_assessment = pd.read_csv(
        f"{data_dir}/studentAssessment.csv",
        dtype={"id_assessment": "int32", "id_student": "int32", "date_submitted": "int32", "is_banked": "int8"},
    )
    student_assessment["score"] = pd.to_numeric(student_assessment["score"], errors="coerce").astype("float32")

    vle = pd.read_csv(
        f"{data_dir}/vle.csv",
        dtype={**cat_cols, "id_site": "int32", "activity_type": "category"},
    )
    vle["week_from"] = pd.to_numeric(vle["week_from"], errors="coerce").astype("Int16")
    vle["week_to"] = pd.to_numeric(vle["week_to"], errors="coerce").astype("Int16")

    # studentVle : 10.6M lignes / 433 Mo -> lecture par chunks, dtypes serres
    vle_dtypes = {
        "code_module": "category", "code_presentation": "category",
        "id_student": "int32", "id_site": "int32",
        "date": "int16", "sum_click": "int16",
    }
    chunks = pd.read_csv(f"{data_dir}/studentVle.csv", dtype=vle_dtypes, chunksize=1_000_000)
    student_vle = pd.concat(chunks, ignore_index=True)

    return {
        "courses": courses,
        "student_info": student_info,
        "student_registration": student_registration,
        "assessments": assessments,
        "student_assessment": student_assessment,
        "vle": vle,
        "student_vle": student_vle,
    }


# ══════════════════════════════════════════════════════════════
# CIBLE BINAIRE — decrochage (Withdrawn=1) vs non-decrochage (0)
# ══════════════════════════════════════════════════════════════

def build_binary_target(student_info: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la colonne 'dropout' : 1=Withdrawn, 0={Pass,Fail,Distinction}."""
    df = student_info.copy()
    df["dropout"] = (df["final_result"] == "Withdrawn").astype("int8")
    return df


# ══════════════════════════════════════════════════════════════
# FUSION IDENTITE (studentInfo + studentRegistration)
# ══════════════════════════════════════════════════════════════

def build_student_table(student_info: pd.DataFrame, student_registration: pd.DataFrame) -> pd.DataFrame:
    keys = ["code_module", "code_presentation", "id_student"]
    merged = student_info.merge(student_registration, on=keys, how="left", validate="one_to_one")
    return merged


# ══════════════════════════════════════════════════════════════
# RAPPORT DE VALIDATION
# ══════════════════════════════════════════════════════════════

def validation_report(tables: dict, student_table: pd.DataFrame) -> None:
    print("=" * 70)
    print("RAPPORT DE VALIDATION — data_loading.py")
    print("=" * 70)

    print("\n--- 1) Formes des tables brutes ---")
    for name, df in tables.items():
        print(f"  {name:22s} shape={df.shape}")

    print("\n--- 2) Valeurs manquantes (colonnes concernees uniquement) ---")
    for name, df in tables.items():
        na = df.isna().sum()
        na = na[na > 0]
        if len(na):
            print(f"  [{name}]")
            for col, n in na.items():
                print(f"      {col:25s} {n:8d} manquants  ({100*n/len(df):.2f}%)")

    print("\n--- 3) Cle primaire (code_module, code_presentation, id_student) ---")
    keys = ["code_module", "code_presentation", "id_student"]
    dup = tables["student_info"].duplicated(subset=keys).sum()
    print(f"  Doublons sur la cle primaire studentInfo : {dup}")
    print(f"  Lignes studentInfo avant fusion : {len(tables['student_info'])}")
    print(f"  Lignes apres fusion avec studentRegistration : {len(student_table)}")
    assert len(student_table) == len(tables["student_info"]), "PERTE DE LIGNES A LA FUSION !"
    print("  -> fusion 1-a-1 confirmee, aucune ligne perdue")

    print("\n--- 4) Coherence des cles etrangeres ---")
    valid_students = set(zip(
        tables["student_info"]["code_module"], tables["student_info"]["code_presentation"],
        tables["student_info"]["id_student"]
    ))
    vle_keys = set(zip(
        tables["student_vle"]["code_module"], tables["student_vle"]["code_presentation"],
        tables["student_vle"]["id_student"]
    ))
    orphans = vle_keys - valid_students
    print(f"  Triplets (module,presentation,id_student) dans studentVle : {len(vle_keys)}")
    print(f"  Triplets absents de studentInfo (orphelins)               : {len(orphans)}")

    print("\n--- 5) Etudiants multi-inscriptions (meme id_student, modules/presentations differents) ---")
    dup_students = tables["student_info"]["id_student"].duplicated().sum()
    print(f"  Etudiants apparaissant plus d'une fois : {dup_students} / {tables['student_info']['id_student'].nunique()} etudiants uniques")
    print("  -> IMPORTANT : le split train/test devra se faire par id_student (pas par ligne)")
    print("     pour eviter qu'un meme etudiant se retrouve a la fois en train et en test.")

    print("\n--- 6) Distribution de la cible binaire (dropout) ---")
    vc = student_table["dropout"].value_counts().sort_index()
    total = len(student_table)
    print(f"  Non-decrochage (0) : {vc[0]:6d}  ({100*vc[0]/total:.1f}%)")
    print(f"  Decrochage    (1) : {vc[1]:6d}  ({100*vc[1]/total:.1f}%)")

    print("\n--- 7) Distribution par module (code_module) ---")
    tab = student_table.groupby("code_module", observed=True)["dropout"].agg(["count", "mean"])
    tab.columns = ["n_etudiants", "taux_decrochage"]
    print(tab.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n--- 8) Volume studentVle (table lourde) ---")
    sv = tables["student_vle"]
    print(f"  Lignes                         : {len(sv):,}")
    print(f"  Etudiants distincts             : {sv['id_student'].nunique()}")
    print(f"  Plage de dates (jours relatifs)  : [{sv['date'].min()}, {sv['date'].max()}]")
    print(f"  Memoire occupee (dtypes optimises): {sv.memory_usage(deep=True).sum() / 1e6:.1f} Mo")

    print("\n" + "=" * 70)
    print("VALIDATION TERMINEE")
    print("=" * 70)


if __name__ == "__main__":
    tables = load_raw_tables()
    tables["student_info"] = build_binary_target(tables["student_info"])
    student_table = build_student_table(tables["student_info"], tables["student_registration"])
    validation_report(tables, student_table)

    # Sauvegarde intermediaire pour les modules suivants (preprocessing.py)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "interim")
    os.makedirs(out_dir, exist_ok=True)
    student_table.to_parquet(f"{out_dir}/student_table.parquet")
    tables["student_vle"].to_parquet(f"{out_dir}/student_vle.parquet")
    tables["vle"].to_parquet(f"{out_dir}/vle.parquet")
    tables["assessments"].to_parquet(f"{out_dir}/assessments.parquet")
    tables["student_assessment"].to_parquet(f"{out_dir}/student_assessment.parquet")
    tables["courses"].to_parquet(f"{out_dir}/courses.parquet")
    print(f"\n[OK] Tables intermediaires sauvegardees dans {out_dir}/")
