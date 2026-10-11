"""T-18, lot 3 : intervalles de confiance des AUC par module et par présentation.
Usage : python eval_module_ci.py [<nombre de rééchantillons, défaut 2000>]
Bootstrap par étudiant À L'INTÉRIEUR de chaque sous-groupe (module ou présentation), sur les prédictions hors pli du
modèle global XGBoost (T-07) ; l'AUC du sous-groupe est calculée sur ses prédictions regroupées, comme dans
results_by_module.json (les sous-groupes sont trop petits pour une AUC par pli). Aucun modèle n'est ré-entraîné.
Sortie : results/early_windows/results_by_module_ci.json. Idempotent (FORCE=1 pour recalculer)."""
import os, sys, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from boot import Boot
from sklearn.metrics import roc_auc_score
OUTJ = f'{RES}/results_by_module_ci.json'
skip_if_exists(OUTJ, 'intervalles par module et par présentation')
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
mod = cohort['code_module'].astype(str).values; pres = cohort['code_presentation'].astype(str).values
res = {'meta': {'n_boot': NB, 'unit': 'id_student, rééchantillonné dans le sous-groupe', 'statistic': 'AUC-ROC sur les prédictions hors pli du sous-groupe', 'seed': SEED}}
for design, w in [('B', 12), ('B', 4), ('A', 39)]:
    z = np.load(f'{ART}/oof/oof_{design}_w{w}.npz'); idx, yy = z['idx'], z['y'].astype(int)
    r = {}
    for i, name in enumerate(z['names']):
        s = z[f'm{i}']; r[str(name)] = {}
        for by, lab in [('by_module', mod[idx]), ('by_presentation', pres[idx])]:
            out = {}
            for g in sorted(set(lab)):
                mk = lab == g; b = Boot(gid[idx][mk], yy[mk], NB, seed=SEED).auc(s[mk])
                out[g] = {'n': int(mk.sum()), 'n_pos': int(yy[mk].sum()), 'auc': float(roc_auc_score(yy[mk], s[mk])), 'ci95': [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
            r[str(name)][by] = out
    res[f'{design}{w}'] = r
    print(f"== {design} w={w} : {len(r)} modèles", flush=True)
    for g, v in r['Raw sequence']['by_module'].items(): print(f"   Raw sequence  {g}  {v['auc']:.4f} [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]  (n={v['n']}, retraits={v['n_pos']})", flush=True)
json.dump(res, open(OUTJ, 'w')); print(f"[OK] {OUTJ}", flush=True)
