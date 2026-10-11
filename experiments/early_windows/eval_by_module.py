"""T-07 : résultats par module et par présentation (conceptions A et B), à partir de prédictions hors pli SAUVEGARDÉES.
Usage : python eval_by_module.py
Étape 1 — prédictions hors pli (GroupKFold(5) par étudiant, XGBoost, mêmes hyperparamètres que eval_windows.py / eval_designA.py),
          une archive par (conception, fenêtre) : interim/early_windows/oof/oof_<A|B>_w<w>.npz  (sautée si elle existe).
          Conception B : étudiants encore inscrits à la fin de la fenêtre (4, 8, 12, 20 sem.) ; autoencodeur à 4, 8, 12 sem.
          Conception A : tous les étudiants (4, 8, 12, 20, 39 sem.) ; à 39 sem. l'embedding v11 (interim/embeddings.npy) est ajouté.
Étape 2 — un modèle GLOBAL est évalué dans chaque sous-groupe : AUC-ROC et PR-AUC par module, par présentation et par
          couple module x présentation -> results/early_windows/results_by_module.json.
'overall' = AUC sur les prédictions hors pli regroupées ; 'fold_auc' = AUC de chacun des 5 plis (plis de CETTE machine : voir R-17,
ils diffèrent de ceux des résultats antérieurs ; l'affectation des plis est conservée dans chaque archive, clé 'fold').
Un AUC vaut null si le sous-groupe ne contient qu'une seule classe. Idempotent (FORCE=1 pour recalculer)."""
import os, sys, json, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
OUTJ = f'{RES}/results_by_module.json'
skip_if_exists(OUTJ, 'résultats par module / présentation')
def XGB(): return xgb.XGBClassifier(n_estimators=200,max_depth=5,learning_rate=0.1,eval_metric='logloss',random_state=SEED,tree_method='hist',n_jobs=1)
OOF = f'{ART}/oof'; os.makedirs(OOF, exist_ok=True)
mod = cohort['code_module'].astype(str).values; pres = cohort['code_presentation'].astype(str).values
modpres = (cohort['code_module'].astype(str)+'_'+cohort['code_presentation'].astype(str)).values
wk_all = seq.sum(2)

def features(design, w):
    """Renvoie (idx, {nom: matrice}) pour une conception et une fenêtre."""
    idx = np.arange(len(y)) if design=='A' else np.where(risk_set(w))[0]
    S = seq[idx][:, :w, :]
    agg = pd.read_parquet(f'{ART}/agg_w{w}.parquet').values[idx]
    F = {'Aggregated(8)': agg, 'Raw sequence': np.log1p(S).reshape(len(idx),-1)}
    emb = None
    if design=='A':
        F['Last active week (1 var.)'] = np.array([np.max(np.where(r[:w]>0)[0]) if (r[:w]>0).any() else -1 for r in wk_all])[:,None].astype('float32')
        if w==39: emb = np.load(f'{UP}/embeddings.npy')
    elif os.path.exists(f'{ART}/ae_w{w}.npz'):
        z = np.load(f'{ART}/ae_w{w}.npz'); assert np.array_equal(z['idx'], idx); emb = z['emb']
    if emb is not None:
        F['AE embedding (64)'] = emb; F['Combined (AE + 8 desc.)'] = np.c_[agg, emb]
    return idx, F

def oof_for(design, w):
    f = f'{OOF}/oof_{design}_w{w}.npz'
    if os.path.exists(f) and not FORCE:
        print(f'[SKIP] prédictions hors pli {design} w={w} existent déjà', flush=True); return f
    idx, F = features(design, w); yy, gg = y[idx], gid[idx]
    fold = np.full(len(idx), -1, dtype='int8'); out = {}
    for k,(tr,te) in enumerate(GroupKFold(5).split(idx,yy,gg)): fold[te] = k
    for name, X in F.items():
        t=time.time(); s = np.full(len(idx), np.nan, dtype='float32')
        for k in range(5):
            tr, te = np.where(fold!=k)[0], np.where(fold==k)[0]
            s[te] = XGB().fit(X[tr],yy[tr]).predict_proba(X[te])[:,1]
        out[name] = s
        print(f"   {design} w={w:2d} {name:28s} AUC par pli={np.mean([roc_auc_score(yy[fold==k],s[fold==k]) for k in range(5)]):.4f} ({time.time()-t:.0f}s)", flush=True)
    np.savez(f, idx=idx, y=yy.astype('int8'), fold=fold, names=np.array(list(out)), **{f'm{i}':v for i,v in enumerate(out.values())})
    return f

def metr(yt, s):
    n, npos = int(len(yt)), int(yt.sum()); ok = 0 < npos < n
    return {'n':n, 'n_pos':npos, 'prev':(float(yt.mean()) if n else None),
            'auc':(float(roc_auc_score(yt,s)) if ok else None), 'ap':(float(average_precision_score(yt,s)) if ok else None)}

res = {'meta': {'cv':'GroupKFold(5) par id_student', 'classifier':'XGBoost (200 arbres, prof. 5, lr 0.1, graine 42)',
                'note':"modèle global évalué dans chaque sous-groupe ; prédictions hors pli dans interim/early_windows/oof/"}, 'A':{}, 'B':{}}
for design, wins in [('B',[4,8,12,20]), ('A',[4,8,12,20,39])]:
    for w in wins:
        z = np.load(oof_for(design, w)); idx, yy, fold = z['idx'], z['y'].astype(int), z['fold']
        m_, p_, mp_ = mod[idx], pres[idx], modpres[idx]
        r = {'N':int(len(idx)), 'prev':float(yy.mean()), 'models':{}}
        for i, name in enumerate(z['names']):
            s = z[f'm{i}']
            r['models'][str(name)] = {
                'overall': metr(yy, s),
                'fold_auc': [float(roc_auc_score(yy[fold==k], s[fold==k])) for k in range(5)],
                'by_module': {g: metr(yy[m_==g], s[m_==g]) for g in sorted(set(m_))},
                'by_presentation': {g: metr(yy[p_==g], s[p_==g]) for g in sorted(set(p_))},
                'by_module_presentation': {g: metr(yy[mp_==g], s[mp_==g]) for g in sorted(set(mp_))}}
        res[design][str(w)] = r
        print(f"== {design} w={w} : {len(r['models'])} modèles, {len(set(m_))} modules, {len(set(p_))} présentations, {len(set(mp_))} couples", flush=True)
json.dump(res, open(OUTJ,'w'))
print(f"[OK] {OUTJ}", flush=True)
