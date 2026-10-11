import os, sys, json, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
w=12
skip_if_exists(f'{RES}/results_extra_labels.json', 'contrôles supplémentaires')
def LR(C=1.0): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=C, class_weight='balanced', random_state=SEED))
z = np.load(f'{ART}/ae_w{w}.npz'); idx=z['idx']; emb=z['emb']; embu=z['emb_untrained']
yy, gg = y[idx], gid[idx]
flat = np.log1p(seq[idx][:, :w, :]).reshape(len(idx),-1)
Xn = norm(seq[idx][:, :w, :]).reshape(len(idx),-1); pc = PCA(n_components=64, random_state=SEED).fit_transform(Xn)
fracs=[0.01,0.02,0.05,0.10,0.25,1.0]; reps={0.01:3,0.02:3,0.05:3,0.10:2,0.25:1,1.0:1}
out={}
def rec(f,name,yt,s): out.setdefault(str(f),{}).setdefault(name,[]).append(roc_auc_score(yt,s))
for fo,(tr,te) in enumerate(GroupKFold(5).split(emb,yy,gg)):
    for f in fracs:
        for r in range(reps[f]):
            rng = np.random.default_rng(1000*fo+r)           # mêmes sous-échantillons que eval_labels.py
            sub = tr if f==1.0 else rng.choice(tr, size=max(int(len(tr)*f),30), replace=False)
            if yy[sub].min()==yy[sub].max(): continue
            rec(f,'AE + LogReg (réf.)', yy[te], LR().fit(emb[sub],yy[sub]).predict_proba(emb[te])[:,1])
            rec(f,'PCA-64 + LogReg', yy[te], LR().fit(pc[sub],yy[sub]).predict_proba(pc[te])[:,1])
            rec(f,'Untrained LSTM enc. + LogReg', yy[te], LR().fit(embu[sub],yy[sub]).predict_proba(embu[te])[:,1])
            rec(f,'Raw sequence + LogReg (C=1)', yy[te], LR(1.0).fit(flat[sub],yy[sub]).predict_proba(flat[te])[:,1])
            rec(f,'Raw sequence + LogReg (C=0.01)', yy[te], LR(0.01).fit(flat[sub],yy[sub]).predict_proba(flat[te])[:,1])
    print("pli",fo+1,flush=True)
json.dump(out, open(f'{RES}/results_extra_labels.json','w'))
# --- cours entier (conception de l'article) : encodeur non entraîné vs autoencodeur de l'article ---
emb39 = np.load(f'{UP}/embeddings.npy'); ae,enc = build_ae(39); X39 = norm(seq); embu39 = enc.predict(X39,batch_size=512,verbose=0)
res39={}
for name,E in [('AE (article)',emb39),('Untrained LSTM encoder',embu39)]:
    a=[];b=[]
    for tr,te in GroupKFold(5).split(E,y,gid):
        a.append(roc_auc_score(y[te], xgb.XGBClassifier(n_estimators=200,max_depth=5,learning_rate=0.1,eval_metric='logloss',random_state=SEED,tree_method='hist',n_jobs=1).fit(E[tr],y[tr]).predict_proba(E[te])[:,1]))
        b.append(roc_auc_score(y[te], LR().fit(E[tr],y[tr]).predict_proba(E[te])[:,1]))
    res39[name]={'xgb':a,'lr':b}; print(name, np.mean(a).round(4), np.mean(b).round(4), flush=True)
json.dump(res39, open(f'{RES}/results_extra_w39.json','w'))
