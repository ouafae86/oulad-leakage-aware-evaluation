"""T-20 : d'où vient l'avantage de la séquence brute sur les descripteurs agrégés ?
Usage : python eval_typetotals.py [<nombre de rééchantillons, défaut 2000>]
Deux contrôles intermédiaires entre les 8 descripteurs et la séquence brute (semaines x 20 types d'activité) :
  C2 weekly totals        : total de clics par semaine (chronologie seule, sans le type d'activité) — déjà dans le tableau 4 ;
  C5 activity-type totals : total de clics par type d'activité sur la fenêtre (détail par type, sans chronologie), 20 variables.
XGBoost, mêmes hyperparamètres et mêmes plis que l'analyse principale (conception B, 4/8/12/20 sem.) ; prédictions hors
pli -> interim/early_windows/oof/oof_B_w<w>_typetotals.npz ; bootstrap apparié par étudiant en moyenne des plis (D-031)
-> results/early_windows/results_typetotals.json. Motif : relecture externe, M-3. Idempotent (FORCE=1 pour recalculer)."""
import os, sys, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from boot import Boot
from sklearn.metrics import roc_auc_score
import xgboost as xgb
OUTJ = f'{RES}/results_typetotals.json'
skip_if_exists(OUTJ, 'contrôle par type d\'activité')
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
OOF = f'{ART}/oof'; M_ = '−'
def XGB(): return xgb.XGBClassifier(n_estimators=200,max_depth=5,learning_rate=0.1,eval_metric='logloss',random_state=SEED,tree_method='hist',n_jobs=1)
def cv(X, yy, fold):
    s = np.full(len(yy), np.nan, dtype='float32')
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]; s[te] = XGB().fit(X[tr], yy[tr]).predict_proba(X[te])[:, 1]
    return s
res = {'meta': {'n_boot': NB, 'statistic': 'AUC-ROC par pli puis moyenne des plis', 'classifier': 'XGBoost (200 arbres, prof. 5, lr 0.1, graine 42)',
                'C5': "log(1 + total de clics par type d'activité sur la fenêtre), 20 variables", 'C2': 'log(1 + total de clics par semaine), w variables'}}
for w in [4, 8, 12, 20]:
    z = np.load(f'{OOF}/oof_B_w{w}.npz'); idx, fold = z['idx'], z['fold']; yy = y[idx]; S_ = seq[idx][:, :w, :]
    f = f'{OOF}/oof_B_w{w}_typetotals.npz'
    if os.path.exists(f) and not FORCE: st = dict(np.load(f))
    else:
        st = {'C5 activity-type totals': cv(np.log1p(S_.sum(1)), yy, fold), 'C2 weekly totals': cv(np.log1p(S_.sum(2)), yy, fold)}; np.savez(f, **st)
    S = {**{k: v for k, v in st.items()}, 'Raw sequence': z[f"m{list(z['names']).index('Raw sequence')}"], 'Aggregated(8)': z[f"m{list(z['names']).index('Aggregated(8)')}"]}
    bt = Boot(gid[idx], yy, NB, seed=SEED); cache = {k: bt.auc_folds(v, fold) for k, v in S.items()}
    fa = lambda s: [float(roc_auc_score(yy[fold == k], s[fold == k])) for k in range(5)]
    r = {'models': {k: {'auc': float(np.mean(fa(v))), 'fold_auc': fa(v), 'ci95': [float(np.percentile(cache[k], 2.5)), float(np.percentile(cache[k], 97.5))]} for k, v in S.items()}, 'comparisons': {}}
    for a_, b_ in [('Raw sequence', 'C5 activity-type totals'), ('C5 activity-type totals', 'Aggregated(8)'), ('C5 activity-type totals', 'C2 weekly totals'), ('Raw sequence', 'C2 weekly totals'), ('C2 weekly totals', 'Aggregated(8)')]:
        d = cache[a_] - cache[b_]
        r['comparisons'][f'{a_} {M_} {b_}'] = {'diff': r['models'][a_]['auc'] - r['models'][b_]['auc'], 'ci95': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    res[str(w)] = r
    print(f"w={w:2d}  " + '  '.join(f"{k.split()[0]}={v['auc']:.4f}" for k, v in r['models'].items()), flush=True)
    for k, v in r['comparisons'].items(): print(f"      {k:48s} {v['diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]", flush=True)
json.dump(res, open(OUTJ, 'w')); print(f"[OK] {OUTJ}", flush=True)
