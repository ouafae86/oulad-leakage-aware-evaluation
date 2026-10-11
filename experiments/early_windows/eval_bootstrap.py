"""T-10 : bootstrap apparié PAR ÉTUDIANT sur les prédictions hors pli (remplace les tests fondés sur 5 plis).
Usage : python eval_bootstrap.py [<nombre de rééchantillons, défaut 2000>]
Entrées (aucun modèle de l'article n'est ré-entraîné) :
  interim/early_windows/oof/oof_<A|B>_w<w>.npz        prédictions hors pli XGBoost (T-07)
  interim/early_windows/b3prime/oof_w12_*_s<g>.npy    B3', B3 rejoué, embedding gelé (T-06)
Étape 1 — prédictions hors pli des contrôles C3 (PCA-64) et C4 (encodeur non entraîné), que eval_windows.py ne sauvegarde
          pas : mêmes plis, même XGBoost -> interim/early_windows/oof/oof_B_w<w>_controls.npz (sautée si elle existe).
Étape 2 — pour chaque comparaison, différence d'AUC-ROC calculée sur les prédictions hors pli regroupées ; les ÉTUDIANTS
          (id_student, toutes leurs inscriptions ensemble) sont rééchantillonnés avec remise ; IC95 % par percentiles.
          Les deux modèles d'une comparaison sont évalués sur le même rééchantillon (apparié).
          Pour un modèle à plusieurs graines, l'AUC est la moyenne des AUC des graines dans chaque rééchantillon.
Portée : l'intervalle reflète l'incertitude d'échantillonnage des étudiants, les modèles étant fixés ; il n'inclut pas
la variabilité d'entraînement (graine, partition), mesurée séparément (T-05, section 11 de SUMMARY.md).
Sortie : results/early_windows/results_bootstrap.json. Idempotent (FORCE=1 pour recalculer)."""
import os, sys, json, time, glob, re; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.metrics import roc_auc_score
OUTJ = f'{RES}/results_bootstrap.json'
skip_if_exists(OUTJ, 'bootstrap apparié par étudiant')
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
OOF = f'{ART}/oof'; CHUNK = 100

def controls(w):
    """Prédictions hors pli de C3 et C4 sur les plis enregistrés dans oof_B_w<w>.npz."""
    f = f'{OOF}/oof_B_w{w}_controls.npz'
    if os.path.exists(f) and not FORCE:
        print(f'[SKIP] contrôles hors pli w={w} existent déjà', flush=True); return f
    import xgboost as xgb
    from sklearn.decomposition import PCA
    z = np.load(f'{OOF}/oof_B_w{w}.npz'); idx, yy, fold = z['idx'], z['y'].astype(int), z['fold']
    a = np.load(f'{ART}/ae_w{w}.npz'); assert np.array_equal(a['idx'], idx)
    Xn = norm(seq[idx][:, :w, :]).reshape(len(idx), -1)
    F = {'C3 PCA-64': PCA(n_components=min(64, Xn.shape[1]-1), random_state=SEED).fit_transform(Xn), 'C4 untrained LSTM encoder': a['emb_untrained']}
    out = {}
    for name, X in F.items():
        s = np.full(len(idx), np.nan, dtype='float32')
        for k in range(5):
            tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
            s[te] = xgb.XGBClassifier(n_estimators=200,max_depth=5,learning_rate=0.1,eval_metric='logloss',random_state=SEED,tree_method='hist',n_jobs=1).fit(X[tr],yy[tr]).predict_proba(X[te])[:,1]
        out[name] = s
        print(f"   contrôle w={w:2d} {name:28s} AUC par pli={np.mean([roc_auc_score(yy[fold==k], s[fold==k]) for k in range(5)]):.4f}", flush=True)
    np.savez(f, idx=idx, names=np.array(list(out)), **{f'm{i}': v for i, v in enumerate(out.values())})
    return f

def load(design, w):
    """Renvoie idx, y, {nom: [scores par graine]} pour une conception et une fenêtre."""
    z = np.load(f'{OOF}/oof_{design}_w{w}.npz'); idx, yy = z['idx'], z['y'].astype(int)
    S = {str(n): [z[f'm{i}']] for i, n in enumerate(z['names'])}
    if design == 'B' and os.path.exists(f'{ART}/ae_w{w}.npz'):
        c = np.load(controls(w)); assert np.array_equal(c['idx'], idx)
        S.update({str(n): [c[f'm{i}']] for i, n in enumerate(c['names'])})
    if design == 'B' and w == 12:
        for tag, name in [('B3prime_pretrained', "B3' pretrained"), ('B3prime_scratch', "B3' scratch"), ('B3_rerun', 'B3 rerun'),
                          ('AE_frozen_XGBoost', 'AE frozen + XGBoost'), ('AE_frozen_LogReg', 'AE frozen + LogReg')]:
            fs = sorted(glob.glob(f'{ART}/b3prime/oof_w12_{tag}_s*.npy'))
            if fs: S[name] = [np.load(f) for f in fs]
    return idx, yy, S

class Boot:
    """AUC pondérée exacte (ex æquo compris) pour NB rééchantillons d'étudiants, par blocs."""
    def __init__(self, idx, yy, nb):
        self.y = yy; self.n = len(yy)
        _, self.g = np.unique(gid[idx], return_inverse=True); ng = self.g.max() + 1
        rng = np.random.default_rng(SEED)
        self.counts = [rng.multinomial(ng, np.full(ng, 1/ng), size=min(CHUNK, nb - b)).astype('float32') for b in range(0, nb, CHUNK)]
        self.cache = {}
    def auc(self, s):
        """Renvoie le vecteur des NB AUC pour un vecteur de scores."""
        key = id(s)
        if key in self.cache: return self.cache[key]
        u, inv = np.unique(s, return_inverse=True); order = np.argsort(inv, kind='stable'); starts = np.searchsorted(inv[order], np.arange(len(u)))
        out = []
        for C in self.counts:
            Wt = C[:, self.g]                                         # poids par ligne (inscription), blocs x N
            P = np.add.reduceat((Wt * self.y)[:, order], starts, axis=1)       # poids positifs par valeur de score
            Ng = np.add.reduceat((Wt * (1 - self.y))[:, order], starts, axis=1)
            below = np.cumsum(Ng, axis=1) - Ng                        # négatifs de score strictement inférieur
            out.append((P * (below + 0.5 * Ng)).sum(1) / (P.sum(1) * Ng.sum(1)))
        self.cache[key] = np.concatenate(out); return self.cache[key]
    def model(self, seeds): return np.mean([self.auc(s) for s in seeds], axis=0)

COMP = {  # (conception, fenêtres) -> comparaisons a − b
    'B': [('AE embedding (64)', 'C4 untrained LSTM encoder'), ('AE embedding (64)', 'C3 PCA-64'), ('AE embedding (64)', 'Raw sequence'),
          ('Combined (AE + 8 desc.)', 'Aggregated(8)'), ('Combined (AE + 8 desc.)', 'Raw sequence'), ('Raw sequence', 'Aggregated(8)'),
          ("B3' pretrained", "B3' scratch"), ("B3' pretrained", 'B3 rerun'), ("B3' scratch", 'B3 rerun'),
          ("B3' pretrained", 'AE frozen + LogReg'), ("B3' pretrained", 'AE frozen + XGBoost'), ("B3' pretrained", 'Raw sequence'), ("B3' scratch", 'Raw sequence')],
    'A': [('Raw sequence', 'Aggregated(8)'), ('Raw sequence', 'Last active week (1 var.)'), ('Aggregated(8)', 'Last active week (1 var.)'),
          ('AE embedding (64)', 'Raw sequence'), ('Combined (AE + 8 desc.)', 'Aggregated(8)'), ('Combined (AE + 8 desc.)', 'Raw sequence')]}
res = {'meta': {'n_boot': NB, 'unit': 'id_student (toutes les inscriptions de l\'étudiant ensemble)', 'ci': 'percentiles 2,5 % et 97,5 %',
                'statistic': 'AUC-ROC sur les prédictions hors pli regroupées ; moyenne des graines pour les modèles à plusieurs graines',
                'scope': "incertitude d'échantillonnage des étudiants, modèles fixés ; n'inclut pas la variabilité d'entraînement", 'seed': SEED}, 'A': {}, 'B': {}}
t0 = time.time()
for design, wins in [('B', [4, 8, 12, 20]), ('A', [4, 8, 12, 20, 39])]:
    for w in wins:
        idx, yy, S = load(design, w); bt = Boot(idx, yy, NB)
        pt = lambda seeds: float(np.mean([roc_auc_score(yy, s) for s in seeds]))
        r = {'N': int(len(idx)), 'n_students': int(bt.g.max() + 1), 'models': {}, 'comparisons': {}}
        for name, seeds in S.items():
            b = bt.model(seeds); r['models'][name] = {'auc': pt(seeds), 'ci95': [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], 'n_seeds': len(seeds)}
        for a, b in COMP[design]:
            if a in S and b in S:
                d = bt.model(S[a]) - bt.model(S[b])
                r['comparisons'][f'{a} − {b}'] = {'diff': pt(S[a]) - pt(S[b]), 'ci95': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                                 'share_positive': float((d > 0).mean())}
        res[design][str(w)] = r
        print(f"== {design} w={w} : {len(S)} modèles, {len(r['comparisons'])} comparaisons, {r['n_students']} étudiants ({time.time()-t0:.0f}s)", flush=True)
        for k, v in r['comparisons'].items(): print(f"   {k:58s} {v['diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]", flush=True)
json.dump(res, open(OUTJ, 'w'))
print(f"[OK] {OUTJ}", flush=True)
