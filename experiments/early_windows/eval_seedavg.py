"""T-20 : comparaisons de l'autoencodeur en MOYENNE SUR CINQ GRAINES (au lieu de la seule graine 42).
Usage : python eval_seedavg.py [<nombre de rééchantillons, défaut 2000>]
Pour chaque fenêtre (4, 8, 12, 20) et chaque graine d'autoencodeur (42 à 46, artefacts seeds/ae_w<w>_s<g>.npz existants,
aucun autoencodeur ré-entraîné) : prédictions hors pli XGBoost, sur les plis de l'analyse principale, pour l'embedding,
l'encodeur non entraîné de même graine et la configuration Combined -> interim/early_windows/oof/oof_B_w<w>_seeds.npz.
Puis bootstrap apparié par étudiant en moyenne des plis (D-031), l'AUC d'une représentation étant la moyenne des AUC
de ses cinq graines dans chaque rééchantillon -> results/early_windows/results_seedavg.json.
Motif : la graine 42 est l'une des plus favorables à l'autoencodeur à 8 et 12 semaines (relecture externe, M-1).
Idempotent (FORCE=1 pour recalculer)."""
import os, sys, json, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from boot import Boot
from sklearn.metrics import roc_auc_score
import xgboost as xgb
OUTJ = f'{RES}/results_seedavg.json'
skip_if_exists(OUTJ, 'comparaisons en moyenne sur cinq graines')
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
SEEDS = [42, 43, 44, 45, 46]; OOF = f'{ART}/oof'
def XGB(): return xgb.XGBClassifier(n_estimators=200,max_depth=5,learning_rate=0.1,eval_metric='logloss',random_state=SEED,tree_method='hist',n_jobs=1)
def cv(X, yy, fold):
    s = np.full(len(yy), np.nan, dtype='float32')
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]; s[te] = XGB().fit(X[tr], yy[tr]).predict_proba(X[te])[:, 1]
    return s
M_ = '−'
res = {'meta': {'ae_seeds': SEEDS, 'n_boot': NB, 'statistic': "AUC-ROC par pli puis moyenne des plis ; moyenne des cinq graines d'autoencodeur", 'classifier': 'XGBoost (200 arbres, prof. 5, lr 0.1, graine 42)'}}
t0 = time.time()
for w in [4, 8, 12, 20]:
    z = np.load(f'{OOF}/oof_B_w{w}.npz'); idx, fold = z['idx'], z['fold']; yy = y[idx]
    c = np.load(f'{OOF}/oof_B_w{w}_controls.npz'); assert np.array_equal(c['idx'], idx)
    agg = pd.read_parquet(f'{ART}/agg_w{w}.parquet').values[idx]
    f = f'{OOF}/oof_B_w{w}_seeds.npz'
    if os.path.exists(f) and not FORCE: st = dict(np.load(f)); print(f'[SKIP] prédictions hors pli par graine w={w} existent déjà', flush=True)
    else:
        st = {}
        for g in SEEDS:
            a = np.load(f'{ART}/seeds/ae_w{w}_s{g}.npz'); assert np.array_equal(a['idx'], idx)
            st[f'emb_s{g}'] = cv(a['emb'], yy, fold); st[f'unt_s{g}'] = cv(a['emb_untrained'], yy, fold); st[f'comb_s{g}'] = cv(np.c_[agg, a['emb']], yy, fold)
            print(f"   w={w:2d} graine {g} : embedding {np.mean([roc_auc_score(yy[fold==k], st[f'emb_s{g}'][fold==k]) for k in range(5)]):.4f}  non entraîné {np.mean([roc_auc_score(yy[fold==k], st[f'unt_s{g}'][fold==k]) for k in range(5)]):.4f}  ({time.time()-t0:.0f}s)", flush=True)
        np.savez(f, **st)
    S = {'Embedding (5 seeds)': [st[f'emb_s{g}'] for g in SEEDS], 'C4 untrained (5 seeds)': [st[f'unt_s{g}'] for g in SEEDS], 'Combined (5 seeds)': [st[f'comb_s{g}'] for g in SEEDS],
         'Raw sequence': [z[f"m{list(z['names']).index('Raw sequence')}"]], 'Aggregated(8)': [z[f"m{list(z['names']).index('Aggregated(8)')}"]], 'C3 PCA-64': [c[f"m{list(c['names']).index('C3 PCA-64')}"]]}
    bt = Boot(gid[idx], yy, NB, seed=SEED); cache = {k: bt.model_folds(v, fold) for k, v in S.items()}
    fa = lambda v: np.mean([[roc_auc_score(yy[fold == k], s[fold == k]) for k in range(5)] for s in v], axis=0)       # AUC par pli, moyenné sur les graines
    r = {'models': {k: {'auc': float(fa(v).mean()), 'fold_auc': [float(x) for x in fa(v)], 'ci95': [float(np.percentile(cache[k], 2.5)), float(np.percentile(cache[k], 97.5))], 'n_seeds': len(v)} for k, v in S.items()}, 'comparisons': {}}
    for a_, b_ in [('Embedding (5 seeds)', 'C4 untrained (5 seeds)'), ('Embedding (5 seeds)', 'C3 PCA-64'), ('Embedding (5 seeds)', 'Raw sequence'), ('Combined (5 seeds)', 'Aggregated(8)'), ('Combined (5 seeds)', 'Raw sequence')]:
        d = cache[a_] - cache[b_]
        r['comparisons'][f'{a_} {M_} {b_}'] = {'diff': r['models'][a_]['auc'] - r['models'][b_]['auc'], 'ci95': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], 'share_positive': float((d > 0).mean())}
        v = r['comparisons'][f'{a_} {M_} {b_}']; print(f"   w={w:2d} {a_} {M_} {b_:24s} {v['diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]", flush=True)
    res[str(w)] = r
json.dump(res, open(OUTJ, 'w')); print(f"[OK] {OUTJ}", flush=True)
