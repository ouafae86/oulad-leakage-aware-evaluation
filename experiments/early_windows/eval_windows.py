import sys, json, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.decomposition import PCA
import xgboost as xgb
def XGB(): return xgb.XGBClassifier(n_estimators=200,max_depth=5,learning_rate=0.1,eval_metric='logloss',random_state=SEED,tree_method='hist',n_jobs=1)
def cv(X, yy, gg, name, store):
    a, p = [], []
    for tr,te in GroupKFold(5).split(X,yy,gg):
        m = XGB().fit(X[tr],yy[tr]); s = m.predict_proba(X[te])[:,1]
        a.append(roc_auc_score(yy[te],s)); p.append(average_precision_score(yy[te],s))
    store[name] = {'auc':a,'ap':p}
    print(f"   {name:28s} AUC={np.mean(a):.4f}±{np.std(a):.4f}  PR-AUC={np.mean(p):.4f}", flush=True)
# OUT_JSON : nom du fichier de résultats (défaut results_windows.json). T-08 écrit la fenêtre 20 complète dans un
# fichier séparé, pour ne pas mélanger dans un même fichier des plis différents (R-17, D-023).
OUTJ = f"{RES}/{os.environ.get('OUT_JSON','results_windows.json')}"
res = json.load(open(OUTJ)) if os.path.exists(OUTJ) else {}
wins = [int(a) for a in sys.argv[1:]] or [4,8,12,20]
for w in wins:
    if str(w) in res and not FORCE:
        print(f'[SKIP] fenêtre {w} déjà évaluée', flush=True); continue
    if w==39:
        idx = np.arange(len(y)); emb = np.load(f'{UP}/embeddings.npy'); embu=None
    else:
        if os.path.exists(f'{ART}/ae_w{w}.npz'):
            z = np.load(f'{ART}/ae_w{w}.npz'); idx=z['idx']; emb=z['emb']; embu=z['emb_untrained']
        else:
            idx = np.where(risk_set(w))[0]; emb=None; embu=None
    yy, gg = y[idx], gid[idx]
    T = min(w,39); S = seq[idx][:, :T, :]
    agg = pd.read_parquet(f'{ART}/agg_w{w}.parquet').values[idx]
    wk = np.log1p(S.sum(2)); flat = np.log1p(S).reshape(len(idx),-1)
    print(f"== fenêtre {w} sem. | N={len(idx)} | taux de retrait futur={yy.mean():.3f}", flush=True)
    r = {'N':int(len(idx)),'prev':float(yy.mean())}
    cv(agg,yy,gg,'Aggregated(8) [B1-like]',r)
    cv(wk,yy,gg,'C2 weekly totals',r)
    cv(flat,yy,gg,'C1 raw sequence flattened',r)
    if w!=39:
        Xn = norm(seq[idx][:, :T, :]).reshape(len(idx),-1)
        pc = PCA(n_components=min(64, Xn.shape[1]-1), random_state=SEED).fit_transform(Xn)
        cv(pc,yy,gg,'C3 PCA-64',r)
        if embu is not None: cv(embu,yy,gg,'C4 untrained LSTM encoder',r)
    if emb is not None:
        cv(emb,yy,gg,'AE embedding (64)',r)
        cv(np.c_[agg,emb],yy,gg,'Combined (AE + 8 desc.)',r)
    res[str(w)] = r
    json.dump(res, open(OUTJ,'w'))
