"""T-18, lot 3 : stabilité des différences clés quand la partition de validation croisée change.
Usage : python eval_partitions.py [<nombre de partitions, défaut 5>]
Conception B, fenêtres 4/8/12/20 : pour R partitions GroupKFold(5, shuffle=True, random_state=r) groupées par étudiant,
on ré-évalue avec XGBoost (mêmes hyperparamètres) les descripteurs agrégés, la séquence brute, la PCA-64, l'encodeur
non entraîné et l'embedding de l'autoencodeur (artefacts existants : aucun autoencodeur n'est ré-entraîné).
Sortie : AUC moyenne des plis par modèle et par partition -> results/early_windows/results_partitions.json.
Ne couvre pas les modèles supervisés séquentiels (B3, B3') : trop coûteux à ré-entraîner sur R partitions.
Idempotent (FORCE=1 pour recalculer)."""
import os, sys, json, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
import xgboost as xgb
OUTJ = f'{RES}/results_partitions.json'
skip_if_exists(OUTJ, 'stabilité selon la partition')
R = int(sys.argv[1]) if len(sys.argv) > 1 else 5
def XGB(): return xgb.XGBClassifier(n_estimators=200,max_depth=5,learning_rate=0.1,eval_metric='logloss',random_state=SEED,tree_method='hist',n_jobs=1)
res = {'meta': {'n_partitions': R, 'cv': 'GroupKFold(5, shuffle=True, random_state=r) par id_student, r = 0..R-1', 'classifier': 'XGBoost (200 arbres, prof. 5, lr 0.1, graine 42)'}}
t0 = time.time()
for w in [4, 8, 12, 20]:
    a = np.load(f'{ART}/ae_w{w}.npz'); idx = a['idx']; yy, gg = y[idx], gid[idx]
    S = seq[idx][:, :w, :]; Xn = norm(S).reshape(len(idx), -1)
    F = {'Aggregated(8)': pd.read_parquet(f'{ART}/agg_w{w}.parquet').values[idx], 'Raw sequence': np.log1p(S).reshape(len(idx), -1),
         'C3 PCA-64': PCA(n_components=min(64, Xn.shape[1]-1), random_state=SEED).fit_transform(Xn), 'C4 untrained LSTM encoder': a['emb_untrained'], 'AE embedding (64)': a['emb']}
    r = {name: [] for name in F}
    for p in range(R):
        folds = list(GroupKFold(5, shuffle=True, random_state=p).split(idx, yy, gg))
        for name, X in F.items():
            r[name].append(float(np.mean([roc_auc_score(yy[te], XGB().fit(X[tr], yy[tr]).predict_proba(X[te])[:, 1]) for tr, te in folds])))
        print(f"w={w:2d} partition {p} : " + '  '.join(f"{n.split()[0]}={r[n][-1]:.4f}" for n in F) + f"  ({time.time()-t0:.0f}s)", flush=True)
    res[str(w)] = r
    json.dump(res, open(OUTJ, 'w'))
print(f"[OK] {OUTJ}", flush=True)
