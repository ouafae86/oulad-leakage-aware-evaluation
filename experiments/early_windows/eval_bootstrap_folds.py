"""D-031 : bootstrap apparié par étudiant avec la statistique « moyenne des plis ».
Usage : python eval_bootstrap_folds.py [<nombre de rééchantillons, défaut 2000>]
Même rééchantillonnage d'étudiants que eval_bootstrap.py (T-10, même graine, mêmes tirages), mais dans chaque rééchantillon
l'AUC-ROC est calculée DANS chacun des 5 plis de test puis moyennée sur les plis, comme les moyennes rapportées dans les
tableaux. L'AUC sur prédictions regroupées (T-10) mélange des scores issus de modèles différents ; elle pénalise les
modèles dont l'échelle des scores varie d'un pli à l'autre (réseaux de neurones) face à XGBoost.
Entrées : prédictions hors pli déjà sauvegardées (T-06, T-07, T-10, T-16) ; aucun modèle n'est ré-entraîné.
Sortie : results/early_windows/results_bootstrap_folds.json (results_bootstrap.json n'est pas modifié). Idempotent."""
import os, sys, json, time, glob; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from boot import Boot
from sklearn.metrics import roc_auc_score
OUTJ = f'{RES}/results_bootstrap_folds.json'
skip_if_exists(OUTJ, 'bootstrap par étudiant, moyenne des plis')
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
OOF = f'{ART}/oof'

def load(design, w):
    """idx, y, plis, {nom: [scores par graine]} pour une conception et une fenêtre."""
    z = np.load(f'{OOF}/oof_{design}_w{w}.npz'); idx, yy, fold = z['idx'], z['y'].astype(int), z['fold']
    S = {str(n): [z[f'm{i}']] for i, n in enumerate(z['names'])}
    cf = f'{OOF}/oof_B_w{w}_controls.npz'
    if design == 'B' and os.path.exists(cf):
        c = np.load(cf); assert np.array_equal(c['idx'], idx); S.update({str(n): [c[f'm{i}']] for i, n in enumerate(c['names'])})
    if design == 'B' and w == 12:
        for tag, name in [('B3prime_pretrained', "B3' pretrained"), ('B3prime_scratch', "B3' scratch"), ('B3_rerun', 'B3 rerun'),
                          ('AE_frozen_XGBoost', 'AE frozen + XGBoost'), ('AE_frozen_LogReg', 'AE frozen + LogReg')]:
            fs = sorted(glob.glob(f'{ART}/b3prime/oof_w12_{tag}_s*.npy'))
            if fs: S[name] = [np.load(f) for f in fs]
    ef = f'{OOF}/oof_A_w39_endcourse.npz'
    if design == 'A' and w == 39 and os.path.exists(ef):                       # cours entier rejoué (T-16)
        e = np.load(ef); assert np.array_equal(e['fold'], fold)
        for n in e.files:
            if n == 'fold' or n.startswith('B3 supervised LSTM s'): continue
            S[n] = [e[n]]
        S['B3 supervised LSTM'] = [e[n] for n in sorted(e.files) if n.startswith('B3 supervised LSTM s')]
    return idx, yy, fold, S

M_ = '−'
COMP = {
    'B': [('AE embedding (64)', 'C4 untrained LSTM encoder'), ('AE embedding (64)', 'C3 PCA-64'), ('AE embedding (64)', 'Raw sequence'),
          ('Combined (AE + 8 desc.)', 'Aggregated(8)'), ('Combined (AE + 8 desc.)', 'Raw sequence'), ('Raw sequence', 'Aggregated(8)'),
          ("B3' pretrained", "B3' scratch"), ("B3' pretrained", 'B3 rerun'), ("B3' scratch", 'B3 rerun'),
          ("B3' pretrained", 'AE frozen + LogReg'), ("B3' pretrained", 'AE frozen + XGBoost'), ("B3' pretrained", 'Raw sequence'), ("B3' scratch", 'Raw sequence'), ('B3 rerun', 'Raw sequence')],
    'A': [('Raw sequence', 'Aggregated(8)'), ('Raw sequence', 'Last active week (1 var.)'), ('Aggregated(8)', 'Last active week (1 var.)'),
          ('AE embedding (64)', 'Raw sequence'), ('Combined (AE + 8 desc.)', 'Aggregated(8)'), ('Combined (AE + 8 desc.)', 'Raw sequence'),
          ('Combined + XGBoost', 'B3 supervised LSTM'), ('Embedding + XGBoost', 'B3 supervised LSTM'), ('Combined + XGBoost', 'Embedding + XGBoost'),
          ('Embedding + XGBoost', 'Aggregated + XGBoost'), ('Combined + XGBoost', 'Raw sequence'), ('B3 supervised LSTM', 'Raw sequence'),
          ('B3 supervised LSTM', 'Last active week (1 var.)'), ('B2 XGBoost + demographics', 'B1 RandomForest'), ('Combined + XGBoost', 'Last active week (1 var.)')]}
res = {'meta': {'n_boot': NB, 'unit': "id_student (toutes les inscriptions de l'étudiant ensemble)", 'ci': 'percentiles 2,5 % et 97,5 %',
                'statistic': "AUC-ROC calculée dans chaque pli de test puis moyennée sur les 5 plis ; moyenne des graines pour les modèles à plusieurs graines",
                'scope': "incertitude d'échantillonnage des étudiants, modèles fixés ; n'inclut pas la variabilité d'entraînement", 'seed': SEED}, 'A': {}, 'B': {}}
t0 = time.time()
for design, wins in [('B', [4, 8, 12, 20]), ('A', [4, 8, 12, 20, 39])]:
    for w in wins:
        idx, yy, fold, S = load(design, w); bt = Boot(gid[idx], yy, NB, seed=SEED); cache = {}
        fm = lambda seeds: float(np.mean([[roc_auc_score(yy[fold == k], s[fold == k]) for k in range(5)] for s in seeds]))
        r = {'N': int(len(idx)), 'n_students': int(bt.g.max() + 1), 'models': {}, 'comparisons': {}}
        for name, seeds in S.items():
            b = bt.model_folds(seeds, fold); cache[name] = b
            r['models'][name] = {'auc': fm(seeds), 'ci95': [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], 'n_seeds': len(seeds)}
        for a, b in COMP[design]:
            if a in S and b in S:
                d = cache[a] - cache[b]
                r['comparisons'][f'{a} {M_} {b}'] = {'diff': fm(S[a]) - fm(S[b]), 'ci95': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], 'share_positive': float((d > 0).mean())}
        res[design][str(w)] = r
        print(f"== {design} w={w} : {len(S)} modèles, {len(r['comparisons'])} comparaisons ({time.time()-t0:.0f}s)", flush=True)
        for k, v in r['comparisons'].items(): print(f"   {k:58s} {v['diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]", flush=True)
json.dump(res, open(OUTJ, 'w'))
print(f"[OK] {OUTJ}", flush=True)
