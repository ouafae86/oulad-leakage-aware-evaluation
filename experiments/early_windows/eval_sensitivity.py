"""T-17 : deux analyses de sensibilité à la définition de la cohorte (XGBoost, mêmes hyperparamètres que eval_windows.py).
Usage : python eval_sensitivity.py
S1 — INCLURE les inscriptions sans aucune activité VLE (exclues de la cohorte d'analyse). Leurs séquences et leurs
     descripteurs valent zéro ; dernière semaine active = -1. Cohorte étendue = cohorte d'analyse + ces inscriptions ;
     GroupKFold(5) par étudiant recalculé sur la cohorte étendue. Conception A à 39 sem. et conception B à 12 sem.
     (ensemble à risque étendu : même règle, désinscription non antérieure au jour 7w, date inconnue => reste).
     Rapporté : AUC de la cohorte étendue, et AUC des seules inscriptions de la cohorte d'analyse dans ces mêmes prédictions.
S2 — EXCLURE les retraits sans date de désinscription des ensembles à risque (conception B, 4/8/12/20 sem.) : mêmes plis
     que l'analyse principale (affectation de oof_B_w<w>.npz), ces inscriptions étant retirées de l'entraînement et du test.
Sortie : results/early_windows/results_sensitivity.json. Idempotent (FORCE=1 pour recalculer)."""
import os, sys, json, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
OUTJ = f'{RES}/results_sensitivity.json'
skip_if_exists(OUTJ, 'analyses de sensibilité (cohorte)')
def XGB(): return xgb.XGBClassifier(n_estimators=200,max_depth=5,learning_rate=0.1,eval_metric='logloss',random_state=SEED,tree_method='hist',n_jobs=1)
def last_active(S):
    wk = S.sum(2); return np.array([np.max(np.where(r > 0)[0]) if (r > 0).any() else -1 for r in wk])[:, None].astype('float32')
def cv(X, yy, fold):
    s = np.full(len(yy), np.nan, dtype='float32')
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]; s[te] = XGB().fit(X[tr], yy[tr]).predict_proba(X[te])[:, 1]
    return s
def summ(yy, s, fold, mask=None):
    """AUC par pli (sur le sous-ensemble mask si fourni), moyenne, PR-AUC regroupée, effectifs."""
    mk = np.ones(len(yy), bool) if mask is None else mask
    fa = [float(roc_auc_score(yy[mk & (fold == k)], s[mk & (fold == k)])) for k in range(5)]
    return {'fold_auc': fa, 'auc': float(np.mean(fa)), 'n': int(mk.sum()), 'n_pos': int(yy[mk].sum())}
res = {'meta': {'classifier': 'XGBoost (200 arbres, prof. 5, lr 0.1, graine 42)', 'cv': 'GroupKFold(5) par id_student'}, 'include_no_vle': {}, 'exclude_no_date': {}}
t0 = time.time()

# ---------------- S1 : inclure les inscriptions sans activité
keys = ['code_module', 'code_presentation', 'id_student']
st = pd.read_parquet(f'{UP}/student_table.parquet')
k_ = lambda d: d[keys].astype(str).agg('|'.join, axis=1)
ex = st[~k_(st).isin(set(k_(cohort)))].reset_index(drop=True)
n0, n1 = len(cohort), len(ex)
y_e = np.r_[y, ex['dropout'].values.astype(int)]; g_e = np.r_[gid, ex['id_student'].values]; u_e = np.r_[unreg, ex['date_unregistration'].astype('float').values]
in_main = np.r_[np.ones(n0, bool), np.zeros(n1, bool)]
seq_e = np.concatenate([seq, np.zeros((n1,) + seq.shape[1:], dtype=seq.dtype)])
res['include_no_vle']['n_added'] = int(n1); res['include_no_vle']['n_added_withdrawn'] = int(ex['dropout'].sum())
for design, w in [('A', 39), ('B', 12)]:
    keep = np.ones(len(y_e), bool) if design == 'A' else ~(u_e < 7*w)
    idx = np.where(keep)[0]; yy, gg, mm = y_e[idx], g_e[idx], in_main[idx]
    fold = np.full(len(idx), -1)
    for k, (tr, te) in enumerate(GroupKFold(5).split(idx, yy, gg)): fold[te] = k
    S = seq_e[idx][:, :w, :]
    agg = np.concatenate([pd.read_parquet(f'{ART}/agg_w{w}.parquet').values, np.zeros((n1, 8))])[idx]
    F = {'Raw sequence': np.log1p(S).reshape(len(idx), -1), 'Aggregated(8)': agg, 'Last active week (1 var.)': last_active(S)}
    r = {'N': int(len(idx)), 'n_added_in_set': int((~mm).sum()), 'n_added_withdrawn_in_set': int(yy[~mm].sum()), 'prev': float(yy.mean()), 'models': {}}
    for name, X in F.items():
        s = cv(X, yy, fold)
        r['models'][name] = {'extended': summ(yy, s, fold), 'analytic_cohort_only': summ(yy, s, fold, mm)}
        print(f"S1 {design} w={w:2d} {name:28s} étendue={r['models'][name]['extended']['auc']:.4f}  cohorte d'analyse seule={r['models'][name]['analytic_cohort_only']['auc']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    res['include_no_vle'][f'{design}{w}'] = r

# ---------------- S2 : exclure les retraits sans date de désinscription (conception B, mêmes plis)
nodate = (y == 1) & np.isnan(unreg)
res['exclude_no_date']['n_no_date'] = int(nodate.sum())
for w in [4, 8, 12, 20]:
    z = np.load(f'{ART}/oof/oof_B_w{w}.npz'); idx0, fold0 = z['idx'], z['fold']
    kp = ~nodate[idx0]; idx, fold = idx0[kp], fold0[kp]; yy = y[idx]
    S = seq[idx][:, :w, :]; agg = pd.read_parquet(f'{ART}/agg_w{w}.parquet').values[idx]
    F = {'Raw sequence': np.log1p(S).reshape(len(idx), -1), 'Aggregated(8)': agg}
    if os.path.exists(f'{ART}/ae_w{w}.npz'):
        a = np.load(f'{ART}/ae_w{w}.npz'); assert np.array_equal(a['idx'], idx0); F['AE embedding (64)'] = a['emb'][kp]
    r = {'N': int(len(idx)), 'n_removed': int((~kp).sum()), 'n_pos': int(yy.sum()), 'n_pos_main': int(y[idx0].sum()), 'models': {}}
    for name, X in F.items():
        s = cv(X, yy, fold); main = z[f"m{list(z['names']).index(name)}"]
        r['models'][name] = {'without_no_date': summ(yy, s, fold), 'main': summ(y[idx0], main, fold0)}
        print(f"S2 B w={w:2d} {name:28s} sans les {r['n_removed']} retraits sans date={r['models'][name]['without_no_date']['auc']:.4f}  analyse principale={r['models'][name]['main']['auc']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    res['exclude_no_date'][str(w)] = r
json.dump(res, open(OUTJ, 'w')); print(f"[OK] {OUTJ}", flush=True)
