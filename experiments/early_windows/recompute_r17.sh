#!/bin/bash
# T-14 / D-026 : recalcul UNIQUE des évaluations de la chaîne précoce qui dépendent de GroupKFold, pour que tous les
# résultats reposent sur la même partition (R-17). Prérequis : accord de l'utilisateur, `make snapshot`, décision D-026.
# - les anciens résultats (plis d'origine) sont d'abord COPIÉS dans results/early_windows/origin_folds/ (jamais écrasés) ;
# - seuls les scripts d'évaluation sont rejoués avec FORCE=1 ; aucun agrégat ni autoencodeur n'est recalculé ;
# - un marqueur empêche tout second passage (le supprimer n'est pas prévu : un nouveau recalcul exige une nouvelle décision).
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-$([ -x ../../.venv/bin/python ] && echo ../../.venv/bin/python || echo python3)}
RES=../../results/early_windows; OLD=$RES/origin_folds; MARK=$OLD/RECALCUL_FAIT.txt
if [ -f "$MARK" ]; then echo "[SKIP] recalcul R-17 déjà effectué ($(cat "$MARK"))"; exit 0; fi
mkdir -p ../../logs/early_windows "$OLD"
LOG="../../logs/early_windows/run_recalcul_R17_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "== recompute_r17.sh $(date) | FORCE=1 sur les évaluations dépendant des plis (D-026)"
for f in results_windows.json results_designA.json results_labels_w12.json results_extra_labels.json results_extra_w39.json; do
  if [ -f "$OLD/$f" ]; then echo "[SKIP] $f déjà conservé dans origin_folds/"; else cp -p "$RES/$f" "$OLD/$f"; echo "conservé : origin_folds/$f"; fi
done
# résultat de T-08 (fenêtre 20 seule, plis du Mac) : redondant après ce recalcul -> déplacé, pas supprimé
if [ -f "$RES/results_windows_w20_ae.json" ]; then mv -n "$RES/results_windows_w20_ae.json" "$OLD/results_windows_w20_ae.json"; echo "déplacé : origin_folds/results_windows_w20_ae.json"; fi
export FORCE=1
$PY eval_windows.py 4 8 12 20
$PY eval_designA.py
$PY eval_labels.py 12
$PY extra_controls.py
$PY eval_by_module.py
echo "recalcul terminé le $(date '+%Y-%m-%d %H:%M:%S') ; journal $(basename "$LOG")" > "$MARK"
echo "[OK] recalcul R-17 terminé"
