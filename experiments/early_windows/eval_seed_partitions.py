"""T-20 : croisement graines d'autoencodeur x partitions de validation croisée.
Usage : python eval_seed_partitions.py [<nombre de partitions, défaut 5>]
Conception B, fenêtres 4/8/12/20. Pour R partitions GroupKFold(5, shuffle=True, random_state=r) (les mêmes que
eval_partitions.py) et les cinq graines d'autoencodeur (42 à 46, artefacts existants), AUC moyenne des plis avec XGBoost
pour l'embedding et l'encodeur non entraîné de même graine, ainsi que pour la séquence brute et la PCA-64.
Sortie : results/early_windows/results_seed_partitions.json (tableaux [partition][graine]).
XGBoost est lancé avec n_jobs=4 (méthode « hist », déterministe) pour limiter la durée ; tous les modèles de ce fichier
sont calculés ici dans les mêmes conditions. Idempotent (FORCE=1 pour recalculer)."""
import os, sys, json, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
import xgboost as xgb
OUTJ = f'{RES}/results_seed_partitions.json'
skip_if_exists(OUTJ, 'croisement graines x partitions')
R = int(sys.argv[1]) if len(sys.argv) > 1 else 5; SEEDS = [42, 43, 44, 45, 46]
def XGB(): return xgb.XGBClassifier(n_estimators=200,max_depth=5,learning_rate=0.1,eval_metric='logloss',random_state=SEED,tree_method='hist',n_jobs=4)
res = {'meta': {'n_partitions': R, 'ae_seeds': SEEDS, 'cv': 'GroupKFold(5, shuffle=True, random_state=r) par id_student, r = 0..R-1', 'classifier': 'XGBoost (200 arbres, prof. 5, lr 0.1, graine 42, n_jobs=4)'}}
t0 = time.time()
for w in [4, 8, 12, 20]:
    idx = np.where(risk_set(w))[0]; yy, gg = y[idx], gid[idx]; S_ = seq[idx][:, :w, :]; Xn = norm(S_).reshape(len(idx), -1)
    A = {g: np.load(f'{ART}/seeds/ae_w{w}_s{g}.npz') for g in SEEDS}; assert all(np.array_equal(A[g]['idx'], idx) for g in SEEDS)
    raw = np.log1p(S_).reshape(len(idx), -1); pca = PCA(n_components=min(64, Xn.shape[1]-1), random_state=SEED).fit_transform(Xn)
    r = {'Embedding': [], 'C4 untrained': [], 'Raw sequence': [], 'C3 PCA-64': []}
    for p in range(R):
        folds = list(GroupKFold(5, shuffle=True, random_state=p).split(idx, yy, gg))
        au = lambda X: float(np.mean([roc_auc_score(yy[te], XGB().fit(X[tr], yy[tr]).predict_proba(X[te])[:, 1]) for tr, te in folds]))
        r['Raw sequence'].append(au(raw)); r['C3 PCA-64'].append(au(pca))
        r['Embedding'].append([au(A[g]['emb']) for g in SEEDS]); r['C4 untrained'].append([au(A[g]['emb_untrained']) for g in SEEDS])
        print(f"w={w:2d} partition {p} : embedding {np.mean(r['Embedding'][-1]):.4f}  non entraîné {np.mean(r['C4 untrained'][-1]):.4f}  brut {r['Raw sequence'][-1]:.4f}  PCA {r['C3 PCA-64'][-1]:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    res[str(w)] = r; json.dump(res, open(OUTJ + '.partiel', 'w'))
json.dump(res, open(OUTJ, 'w')); os.remove(OUTJ + '.partiel'); print(f"[OK] {OUTJ}", flush=True)
