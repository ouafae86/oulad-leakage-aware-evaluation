"""T-05 : variance de l'autoencodeur entre graines (conception B, mêmes plis GroupKFold(5) que eval_windows.py).
Usage : python eval_seeds.py <fenêtre> <graine> [<graine> ...]
Pour chaque graine : AE + XGBoost, AE + LogReg, et l'encodeur non entraîné de la même graine (C4) + XGBoost / LogReg.
Les classifieurs gardent la graine 42 : seule la graine de l'autoencodeur varie.
'reference' = artefact d'origine ae_w<w>.npz, réévalué ici pour vérifier la reproduction de results_windows.json.
Idempotent : une graine déjà présente dans results_seeds_w<w>.json est sautée (FORCE=1 pour recalculer)."""
import sys, json, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
def XGB(): return xgb.XGBClassifier(n_estimators=200,max_depth=5,learning_rate=0.1,eval_metric='logloss',random_state=SEED,tree_method='hist',n_jobs=1)
def LR(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED))
w = int(sys.argv[1]); seeds = [int(a) for a in sys.argv[2:]]
path = f'{RES}/results_seeds_w{w}.json'
res = json.load(open(path)) if os.path.exists(path) else {}
def cv(X, yy, gg, make, name, store):
    a, p = [], []
    for tr,te in GroupKFold(5).split(X,yy,gg):
        s = make().fit(X[tr],yy[tr]).predict_proba(X[te])[:,1]
        a.append(roc_auc_score(yy[te],s)); p.append(average_precision_score(yy[te],s))
    store[name] = {'auc':a,'ap':p}
    print(f"   {name:34s} AUC={np.mean(a):.4f}±{np.std(a):.4f}  PR-AUC={np.mean(p):.4f}", flush=True)
ref_idx = np.where(risk_set(w))[0]
for key, f in [('reference', f'{ART}/ae_w{w}.npz')] + [(str(s), f'{ART}/seeds/ae_w{w}_s{s}.npz') for s in seeds]:
    if key in res and not FORCE:
        print(f'[SKIP] {key} déjà évalué (fenêtre {w})', flush=True); continue
    if not os.path.exists(f):
        print(f'[ABSENT] {f} : lancer train_ae_seeds.py {w} 30 {key}', flush=True); sys.exit(1)
    z = np.load(f); idx = z['idx']
    assert np.array_equal(idx, ref_idx), "ensemble à risque différent : plis non comparables"
    yy, gg = y[idx], gid[idx]
    print(f"== fenêtre {w} sem. | {key} | N={len(idx)} | époques={len(z['loss'])} | val_loss min={z['val_loss'].min():.4f}", flush=True)
    r = {'N':int(len(idx)), 'epochs':int(len(z['loss'])), 'val_loss_min':float(z['val_loss'].min()), 'artefact':os.path.relpath(f, ROOT)}
    cv(z['emb'],yy,gg,XGB,'AE embedding (64) + XGBoost',r)
    cv(z['emb'],yy,gg,LR,'AE embedding (64) + LogReg',r)
    cv(z['emb_untrained'],yy,gg,XGB,'C4 untrained LSTM encoder + XGBoost',r)
    cv(z['emb_untrained'],yy,gg,LR,'C4 untrained LSTM encoder + LogReg',r)
    res[key] = r
    json.dump(res, open(path,'w'))
