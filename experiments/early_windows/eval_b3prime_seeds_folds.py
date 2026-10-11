"""T-20 : AUC par pli des modèles B3' du T-19 (pré-entraînés avec cinq graines d'autoencodeur, et entraînés de zéro),
calculées à partir des prédictions hors pli déjà sauvegardées : aucun modèle n'est ré-entraîné.
Sert à donner, en plus de l'intervalle bootstrap, l'intervalle fondé sur les cinq plis (variance corrigée de Nadeau & Bengio).
Sortie : results/early_windows/results_b3prime_seeds_folds_w12.json. Idempotent (FORCE=1 pour recalculer)."""
import os, sys, json, glob; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.metrics import roc_auc_score
w = 12; D = f'{ART}/b3prime'; OUTJ = f'{RES}/results_b3prime_seeds_folds_w{w}.json'
skip_if_exists(OUTJ, "AUC par pli des B3' multi-graines")
z = np.load(f'{ART}/oof/oof_B_w{w}.npz'); idx, fold = z['idx'], z['fold']; yy = y[idx]
fa = lambda s: [float(roc_auc_score(yy[fold == k], s[fold == k])) for k in range(5)]
avg = lambda fs: [float(v) for v in np.mean([fa(np.load(f)) for f in fs], axis=0)]
res = {'meta': {'window': w, 'statistic': "AUC-ROC par pli, moyennée sur les graines supervisées (et sur les graines d'autoencodeur pour 'all')"}, 'ae': {}}
sc = sorted(glob.glob(f'{D}/oof_w{w}_B3prime_scratch_s*.npy')); res['scratch'] = {'fold_auc': avg(sc), 'n_runs': len(sc)}
allf = []
for a in [42, 43, 44, 45, 46]:
    fs = sorted(glob.glob(f'{D}/oof_w{w}_B3prime_pretrained_s*.npy')) if a == SEED else sorted(glob.glob(f'{D}/oof_w{w}_B3prime_pretrained_ae{a}_s*.npy'))
    assert len(fs) == 3, (a, fs); allf += fs; res['ae'][str(a)] = {'fold_auc': avg(fs), 'n_runs': len(fs)}
res['all_ae_seeds'] = {'fold_auc': avg(allf), 'n_runs': len(allf)}
json.dump(res, open(OUTJ, 'w')); print({k: round(float(np.mean(v['fold_auc'])), 4) for k, v in {**res['ae'], 'all': res['all_ae_seeds'], 'scratch': res['scratch']}.items()}); print(f"[OK] {OUTJ}")
