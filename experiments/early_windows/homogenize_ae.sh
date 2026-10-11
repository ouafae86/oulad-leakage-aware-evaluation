#!/bin/bash
# T-18 / D-033 : opération UNIQUE. L'autoencodeur principal de chaque fenêtre devient celui entraîné sur cette machine
# avec la graine 42 (seeds/ae_w<w>_s42.npz, même protocole) ; les anciens artefacts sont DÉPLACÉS dans superseded/
# (jamais supprimés) ; toutes les évaluations qui lisent ae_w<w>.npz sont rejouées avec FORCE=1.
# Prérequis : accord de l'utilisateur, `make snapshot`, décision D-033. Un marqueur empêche tout second passage.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-$([ -x ../../.venv/bin/python ] && echo ../../.venv/bin/python || echo python3)}
ART=../../interim/early_windows; OLD=$ART/superseded; MARK=$OLD/HOMOGENEISATION_FAITE.txt
if [ -f "$MARK" ]; then echo "[SKIP] homogénéisation D-033 déjà effectuée ($(cat "$MARK"))"; exit 0; fi
mkdir -p ../../logs/early_windows "$OLD"
LOG="../../logs/early_windows/run_homogeneisation_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "== homogenize_ae.sh $(date) | D-033"
# 1. graines manquantes à 8 semaines (les fenêtres 4, 12 et 20 existent déjà)
for s in 42 43 44 45 46; do $PY train_ae_seeds.py 8 30 $s; done
# 2. artefact principal = graine 42 entraînée ici ; ancien artefact déplacé
for w in 4 8 12 20; do
  [ -f "$ART/seeds/ae_w${w}_s42.npz" ] || { echo "[ABSENT] seeds/ae_w${w}_s42.npz"; exit 1; }
  if [ -f "$OLD/ae_w${w}.npz" ]; then echo "[SKIP] ancien ae_w${w}.npz déjà déplacé"; else mv -n "$ART/ae_w${w}.npz" "$OLD/ae_w${w}.npz"; echo "déplacé : superseded/ae_w${w}.npz"; fi
  cp -p "$ART/seeds/ae_w${w}_s42.npz" "$ART/ae_w${w}.npz"; echo "artefact principal w=${w} = seeds/ae_w${w}_s42.npz"
done
# 3. évaluations qui dépendent des autoencodeurs par fenêtre
export FORCE=1
$PY eval_windows.py 4 8 12 20
$PY eval_labels.py 12
$PY eval_lopo.py 12
$PY extra_controls.py
for w in 4 8 12 20; do $PY eval_seeds.py $w 42 43 44 45 46; done
$PY eval_by_module.py
$PY eval_bootstrap.py 2000
$PY eval_bootstrap_folds.py 2000
$PY eval_sensitivity.py
echo "homogénéisation terminée le $(date '+%Y-%m-%d %H:%M:%S') ; journal $(basename "$LOG")" > "$MARK"
echo "[OK] homogénéisation D-033 terminée"
