import os, sys, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
def XGB(): return xgb.XGBClassifier(n_estimators=200,max_depth=5,learning_rate=0.1,eval_metric='logloss',random_state=SEED,tree_method='hist',n_jobs=1)
res={}
skip_if_exists(f'{RES}/results_designA.json', 'conception A')
wk_all = seq.sum(2)
for w in [4,8,12,20,39]:
    S = seq[:, :w, :]; wk = np.log1p(S.sum(2)); flat = np.log1p(S).reshape(len(y),-1)
    agg = pd.read_parquet(f'{ART}/agg_w{w}.parquet').values
    lastact = np.array([np.max(np.where(r[:w]>0)[0]) if (r[:w]>0).any() else -1 for r in wk_all])
    r={}
    for name,X in [('Aggregated(8)',agg),('Weekly totals',wk),('Raw sequence',flat),('Last active week (1 var.)',lastact[:,None].astype('float32'))]:
        a=[];p=[]
        for tr,te in GroupKFold(5).split(X,y,gid):
            s = XGB().fit(X[tr],y[tr]).predict_proba(X[te])[:,1]; a.append(roc_auc_score(y[te],s)); p.append(average_precision_score(y[te],s))
        r[name]={'auc':a,'ap':p}; print(w,name,round(np.mean(a),4),flush=True)
    # part des retraits DÉJÀ connus à la fin de la fenêtre (désinscription avant le jour 7w)
    r['share_already_withdrawn']=float(((unreg<7*w)&(y==1)).sum()/(y==1).sum()); r['N']=int(len(y)); r['prev']=float(y.mean())
    res[str(w)]=r; json.dump(res,open(f'{RES}/results_designA.json','w'))
