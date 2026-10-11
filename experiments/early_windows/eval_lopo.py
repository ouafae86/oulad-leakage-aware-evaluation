import os, sys, json, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from b3 import b3_fit_predict
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
w = int(sys.argv[1])
skip_if_exists(f'{RES}/results_lopo_w{w}.json', f'LOPO w={w}')
def XGB(): return xgb.XGBClassifier(n_estimators=200,max_depth=5,learning_rate=0.1,eval_metric='logloss',random_state=SEED,tree_method='hist',n_jobs=1)
def LR(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED))
z = np.load(f'{ART}/ae_w{w}.npz'); idx=z['idx']; emb=z['emb']
yy, gg, pp = y[idx], gid[idx], cohort['code_presentation'].astype(str).values[idx]
agg = pd.read_parquet(f'{ART}/agg_w{w}.parquet').values[idx]
flat = np.log1p(seq[idx][:, :w, :]).reshape(len(idx),-1); Xs = norm(seq[idx][:, :w, :])
out = {}; t0=time.time()
for P in sorted(set(pp)):
    te = np.where(pp==P)[0]; test_ids = set(gg[te])
    tr = np.where((pp!=P) & ~np.isin(gg, list(test_ids)))[0]        # aucun étudiant du test dans l'entraînement
    r = {'n_train':int(len(tr)),'n_test':int(len(te)),'prev_test':float(yy[te].mean())}
    def add(name, s): r[name] = {'auc':float(roc_auc_score(yy[te],s)),'ap':float(average_precision_score(yy[te],s))}
    add('AE + LogReg',  LR().fit(emb[tr],yy[tr]).predict_proba(emb[te])[:,1])
    add('AE + XGBoost', XGB().fit(emb[tr],yy[tr]).predict_proba(emb[te])[:,1])
    add('Combined (AE+8) + XGBoost', XGB().fit(np.c_[agg,emb][tr],yy[tr]).predict_proba(np.c_[agg,emb][te])[:,1])
    add('Aggregated(8) + XGBoost', XGB().fit(agg[tr],yy[tr]).predict_proba(agg[te])[:,1])
    add('Raw sequence + XGBoost', XGB().fit(flat[tr],yy[tr]).predict_proba(flat[te])[:,1])
    add('B3 supervised LSTM', b3_fit_predict(Xs[tr],yy[tr],Xs[te]))
    out[P]=r; print(P, {k:round(v['auc'],4) for k,v in r.items() if isinstance(v,dict)}, f"({time.time()-t0:.0f}s)", flush=True)
    json.dump(out, open(f'{RES}/results_lopo_w{w}.json','w'))
