import sys, json, os, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from b3 import b3_fit_predict
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
w = int(sys.argv[1])
skip_if_exists(f'{RES}/results_labels_w{w}.json', f'efficacité en étiquettes w={w}')
def XGB(): return xgb.XGBClassifier(n_estimators=200,max_depth=5,learning_rate=0.1,eval_metric='logloss',random_state=SEED,tree_method='hist',n_jobs=1)
def LR(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED))
z = np.load(f'{ART}/ae_w{w}.npz'); idx=z['idx']; emb=z['emb']
yy, gg = y[idx], gid[idx]
agg = pd.read_parquet(f'{ART}/agg_w{w}.parquet').values[idx]
flat = np.log1p(seq[idx][:, :w, :]).reshape(len(idx),-1)
Xs = norm(seq[idx][:, :w, :])
fracs = [0.01,0.02,0.05,0.10,0.25,1.0]
reps = {0.01:3,0.02:3,0.05:3,0.10:2,0.25:1,1.0:1}
b3reps = {0.01:2,0.05:2,0.25:1,1.0:1}
out = {}
def rec(f, name, yt, s):
    out.setdefault(str(f),{}).setdefault(name,{'auc':[],'ap':[]})
    out[str(f)][name]['auc'].append(roc_auc_score(yt,s)); out[str(f)][name]['ap'].append(average_precision_score(yt,s))
t0=time.time()
for fo,(tr,te) in enumerate(GroupKFold(5).split(emb,yy,gg)):
    for f in fracs:
        for r in range(reps[f]):
            rng = np.random.default_rng(1000*fo+r)
            sub = tr if f==1.0 else rng.choice(tr, size=max(int(len(tr)*f),30), replace=False)
            if yy[sub].min()==yy[sub].max(): continue
            rec(f,'AE + LogReg',  yy[te], LR().fit(emb[sub],yy[sub]).predict_proba(emb[te])[:,1])
            rec(f,'AE + XGBoost', yy[te], XGB().fit(emb[sub],yy[sub]).predict_proba(emb[te])[:,1])
            rec(f,'Aggregated(8) + XGBoost', yy[te], XGB().fit(agg[sub],yy[sub]).predict_proba(agg[te])[:,1])
            rec(f,'Raw sequence + XGBoost', yy[te], XGB().fit(flat[sub],yy[sub]).predict_proba(flat[te])[:,1])
            if f in b3reps and r < b3reps[f]:
                rec(f,'B3 supervised LSTM', yy[te], b3_fit_predict(Xs[sub],yy[sub],Xs[te],seed=SEED+r))
    print(f"pli {fo+1}/5 terminé ({time.time()-t0:.0f}s)", flush=True)
    json.dump(out, open(f'{RES}/results_labels_w{w}.json','w'))
