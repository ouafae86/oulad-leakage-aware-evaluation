#!/bin/bash
# Rejoue la chaîne d'expériences précoces (conception B, contrôles, étiquettes rares, LOPO).
# IDEMPOTENT : chaque étape SAUTE si son résultat existe déjà (FORCE=1 pour recalculer). Rien n'est écrasé par défaut.
# Journal : logs/early_windows/run_<horodatage>.log (ajout, les journaux historiques ne sont jamais tronqués).
set -e
cd "$(dirname "$0")"
# interpréteur : $PYTHON, sinon le .venv du projet s'il existe (TensorFlow n'est utilisable que dans le .venv : R-13/R-16), sinon python3
PY=${PYTHON:-$([ -x ../../.venv/bin/python ] && echo ../../.venv/bin/python || echo python3)}
mkdir -p ../../logs/early_windows
LOG="../../logs/early_windows/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "== run_all.sh $(date) | FORCE=${FORCE:-0}"
$PY make_agg.py
for w in 4 8 12 20; do $PY train_ae.py $w 30; done      # 20 sem. ajouté par T-08 (D-024)
$PY eval_windows.py 4 8 12 20
$PY eval_designA.py
$PY eval_labels.py 12
$PY eval_lopo.py 12
$PY extra_controls.py
# T-05 : variance de l'autoencodeur à 12 sem. (5 graines), mêmes plis
for s in 42 43 44 45 46; do $PY train_ae_seeds.py 12 30 $s; done
$PY eval_seeds.py 12 42 43 44 45 46
# T-15, T-18 : même mesure aux fenêtres 4, 8 et 20 sem.
for w in 4 8 20; do for s in 42 43 44 45 46; do $PY train_ae_seeds.py $w 30 $s; done; $PY eval_seeds.py $w 42 43 44 45 46; done
# T-06 : B3' (encodeur 128->64 identique à l'autoencodeur, entraîné de bout en bout), 3 graines
$PY eval_b3prime.py 12 42 43 44
# T-07 : prédictions hors pli sauvegardées + résultats par module et par présentation (conceptions A et B)
$PY eval_by_module.py
# T-16 : B1, B2, B3 et ablation « cours entier » sur la partition actuelle (modèles de src/ importés, non exécutés)
$PY eval_endcourse.py 42 43 44
# T-10 : bootstrap apparié par étudiant sur les prédictions hors pli
$PY eval_bootstrap.py 2000
# D-031 : même bootstrap avec la statistique « moyenne des plis » (celle que cite l'article)
$PY eval_bootstrap_folds.py 2000
# T-17 : sensibilité à la définition de la cohorte (inscriptions sans activité incluses ; retraits sans date exclus)
$PY eval_sensitivity.py
# T-18 : intervalles par module et par présentation ; stabilité des différences clés sur 5 partitions
$PY eval_module_ci.py 2000
$PY eval_partitions.py 5
# T-19 : pré-entraînement de B3' avec 5 graines d'autoencodeur (12 sem.)
$PY eval_b3prime_seeds.py 12 42 43 44 45 46
# T-20 : comparaisons en moyenne sur cinq graines ; contrôle par type d'activité ; croisement graines x partitions
$PY eval_seedavg.py 2000
$PY eval_typetotals.py 2000
$PY eval_seed_partitions.py 5
$PY eval_b3prime_seeds_folds.py
echo "[OK] chaîne terminée (étapes déjà faites sautées)"
