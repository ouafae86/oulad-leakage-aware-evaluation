"""
run_pipeline.py
─────────────────────────────────────────────────────────────────
Orchestrateur unique du pipeline complet.

Execute, dans l'ordre, les 7 modules du pipeline. Chaque etape lit
les fichiers intermediaires (interim/) produits par l'etape
precedente — aucune etape ne recalcule ce qu'une etape anterieure
a deja produit (cf. principe rappele dans chaque module).

Usage :
    python3 run_pipeline.py             # run complet (production)
    python3 run_pipeline.py --quick     # run rapide (validation du code, PAS les resultats finaux)
    python3 run_pipeline.py --skip data_loading,preprocessing   # reprendre a partir d'une etape
─────────────────────────────────────────────────────────────────
"""

import argparse
import subprocess
import sys
import time

STEPS = [
    ("data_loading",         "data_loading.py",         []),
    ("preprocessing",        "preprocessing.py",        []),
    ("features_aggregated",  "features_aggregated.py",  []),
    ("lstm_autoencoder",     "lstm_autoencoder.py",      ["--quick"]),   # flag quick ajoute conditionnellement
    ("baselines",            "baselines.py",            ["--quick"]),
    ("evaluation",           "evaluation.py",           ["--quick"]),
    ("explainability",       "explainability.py",       []),
    ("figures",              "figures.py",               []),
]


def run_step(script: str, extra_args: list):
    cmd = [sys.executable, script] + extra_args
    print("\n" + "#" * 70)
    print(f"# {' '.join(cmd)}")
    print("#" * 70)
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    print(f"[Duree] {script} : {elapsed:.1f}s")
    if result.returncode != 0:
        print(f"[ERREUR] {script} a echoue (code {result.returncode}) — pipeline interrompu.")
        sys.exit(result.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                         help="Run rapide : peu d'epochs / peu de folds, VALIDATION DU CODE UNIQUEMENT")
    parser.add_argument("--skip", type=str, default="",
                         help="Liste d'etapes a sauter, separees par des virgules (ex: data_loading,preprocessing)")
    args = parser.parse_args()
    skip = set(args.skip.split(",")) if args.skip else set()

    t_start = time.time()
    for name, script, quick_args in STEPS:
        if name in skip:
            print(f"[Saute] {name}")
            continue
        extra_args = quick_args if args.quick else []
        run_step(script, extra_args)

    total = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"PIPELINE TERMINE en {total/60:.1f} minutes")
    if args.quick:
        print("[ATTENTION] Mode --quick : ces resultats ne sont PAS les resultats finaux de l'article.")
        print("            Relancer sans --quick pour l'entrainement complet.")
    print("=" * 70)
