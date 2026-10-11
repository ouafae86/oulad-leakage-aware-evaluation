"""T-16 : B1, B2, B3 et l'ablation « cours entier » (39 sem., tous les étudiants) rejoués sur la partition actuelle.
Usage : python eval_endcourse.py [<graine B3> ...]      (défaut 42 43 44)
Les définitions des modèles sont celles du pipeline v11, importées de src/ EN LECTURE SEULE (aucun script de src/
n'est exécuté, rien n'est écrit dans interim/ hors du dossier oof/) :
  B1 : forêt aléatoire sur les 8 descripteurs (src/baselines.py)
  B2 : XGBoost sur descripteurs + démographie, scale_pos_weight (src/baselines.py)
  B3 : LSTM supervisé (Masking -> LSTM(64) -> sigmoïde, <= 15 époques, validation = 10 % derniers du pli d'entraînement)
  ablation : Aggregated / Embedding / Combined x LogisticRegression / RandomForest / XGBoost (src/evaluation.py)
Partition : celle enregistrée dans interim/early_windows/oof/oof_A_w39.npz (identique à GroupKFold(5) recalculé ici).
Sorties : prédictions hors pli -> interim/early_windows/oof/oof_A_w39_endcourse.npz ;
          AUC par pli + bootstrap apparié par étudiant -> results/early_windows/results_endcourse_w39.json.
Idempotent : chaque modèle déjà présent dans l'archive de prédictions est sauté (FORCE=1 pour recalculer)."""
import os, sys, json, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from boot import Boot
sys.path.insert(0, str(ROOT/'src'))
from features_aggregated import AGGREGATED_FEATURE_NAMES
from baselines import DEMOGRAPHIC_COLS, RANDOM_SEED
from evaluation import CLASSIFIERS, build_feature_configs
from lstm_autoencoder import normalize_sequences
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, average_precision_score
OUTJ = f'{RES}/results_endcourse_w39.json'; OOFP = f'{ART}/oof/oof_A_w39_endcourse.npz'
skip_if_exists(OUTJ, 'cours entier sur la partition actuelle (B1, B2, B3, ablation)')
seeds = [int(a) for a in sys.argv[1:]] or [42, 43, 44]
NB = 2000
feats = pd.read_parquet(f'{UP}/aggregated_features.parquet'); emb = np.load(f'{UP}/embeddings.npy')
ref = np.load(f'{ART}/oof/oof_A_w39.npz'); fold = ref['fold']
chk = np.full(len(y), -1)
for k, (tr, te) in enumerate(GroupKFold(5).split(emb, y, gid)): chk[te] = k
assert np.array_equal(chk, fold) and np.array_equal(ref['y'].astype(int), y), "la partition enregistrée ne correspond pas à GroupKFold(5)"
store = dict(np.load(OOFP)) if os.path.exists(OOFP) and not FORCE else {}
store.pop('fold', None)
def save(): np.savez(OOFP, fold=fold, **store)
def run(name, fit):
    """fit(tr, te) -> scores du pli de test. Prédictions hors pli conservées sous la clé `name`."""
    if name in store: print(f'[SKIP] {name} déjà calculé', flush=True); return
    t = time.time(); s = np.full(len(y), np.nan, dtype='float32')
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]; s[te] = fit(tr, te)
    store[name] = s; save()
    print(f"   {name:34s} AUC={np.mean([roc_auc_score(y[fold==k], s[fold==k]) for k in range(5)]):.4f}  ({time.time()-t:.0f}s)", flush=True)

# --- B1, B2 (mêmes constructions que src/baselines.py)
Xagg = feats[AGGREGATED_FEATURE_NAMES]
run('B1 RandomForest', lambda tr, te: RandomForestClassifier(n_estimators=200, max_depth=12, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1).fit(Xagg.iloc[tr], y[tr]).predict_proba(Xagg.iloc[te])[:, 1])
demo = cohort[DEMOGRAPHIC_COLS].copy()
Xb2 = pd.concat([feats[AGGREGATED_FEATURE_NAMES].reset_index(drop=True), demo.reset_index(drop=True)], axis=1)
cat_cols = [c for c in DEMOGRAPHIC_COLS if demo[c].dtype.name in ('category', 'object')]
def b2(tr, te):
    import xgboost as xgb
    pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)], remainder='passthrough')
    clf = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, eval_metric='logloss', random_state=RANDOM_SEED, scale_pos_weight=(y == 0).sum() / (y == 1).sum())
    return Pipeline([('pre', pre), ('clf', clf)]).fit(Xb2.iloc[tr], y[tr]).predict_proba(Xb2.iloc[te])[:, 1]
run('B2 XGBoost + demographics', b2)
# --- ablation (mêmes classifieurs et mêmes matrices que src/evaluation.py)
for cfg, Xc in build_feature_configs(feats, emb).items():
    for mdl, build in CLASSIFIERS.items():
        run(f'{cfg} + {mdl}', lambda tr, te, Xc=Xc, build=build: build().fit(Xc[tr], y[tr]).predict_proba(Xc[te])[:, 1])
# --- B3 (même réseau et même protocole que src/baselines.py ; plusieurs graines)
normalized, _, _ = normalize_sequences(seq)
def b3(seed):
    def fit(tr, te):
        import tensorflow as tf
        from tensorflow.keras import layers, models, callbacks
        tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
        i = layers.Input(shape=(seq.shape[1], seq.shape[2])); x = layers.Masking()(i); x = layers.LSTM(64)(x); o = layers.Dense(1, activation='sigmoid')(x)
        mdl = models.Model(i, o); mdl.compile(optimizer='adam', loss='binary_crossentropy')
        mdl.fit(normalized[tr], y[tr], epochs=15, batch_size=128, verbose=0, validation_split=0.1,
                callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])
        return mdl.predict(normalized[te], batch_size=512, verbose=0).ravel()
    return fit
for sd in seeds: run(f'B3 supervised LSTM s{sd}', b3(sd))

# --- résultats : AUC par pli, AUC sur prédictions regroupées, bootstrap apparié par étudiant
names = [k for k in store if k != 'fold']
groups_ = {n: [n] for n in names if not n.startswith('B3 supervised LSTM s')}; groups_['B3 supervised LSTM'] = [n for n in names if n.startswith('B3 supervised LSTM s')]
for k in ['Raw sequence', 'Last active week (1 var.)']:                       # références déjà calculées sur la même partition (T-07)
    store_ref = ref[f"m{list(ref['names']).index(k)}"]; store[f'ref::{k}'] = store_ref; groups_[k] = [f'ref::{k}']
bt = Boot(gid, y, NB, seed=SEED)
res = {'meta': {'window': 39, 'design': 'A', 'cv': 'GroupKFold(5) par id_student, partition de oof_A_w39.npz', 'n_boot': NB, 'b3_seeds': seeds,
                'source_models': 'src/baselines.py et src/evaluation.py (définitions importées, scripts non exécutés)'}, 'N': int(len(y)), 'models': {}, 'comparisons': {}}
bcache = {}
for g, ks in groups_.items():
    fa = np.mean([[roc_auc_score(y[fold == k], store[n][fold == k]) for k in range(5)] for n in ks], axis=0)
    b = bt.model([store[n] for n in ks]); bcache[g] = b
    res['models'][g] = {'fold_auc': [float(v) for v in fa], 'n_seeds': len(ks), 'auc_pooled': float(np.mean([roc_auc_score(y, store[n]) for n in ks])),
                        'ap_pooled': float(np.mean([average_precision_score(y, store[n]) for n in ks])), 'ci95': [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
    if len(ks) > 1: res['models'][g]['fold_mean_by_seed'] = {n.split(' s')[-1]: float(np.mean([roc_auc_score(y[fold == k], store[n][fold == k]) for k in range(5)])) for n in ks}
for a, b in [('Combined + XGBoost', 'B3 supervised LSTM'), ('Embedding + XGBoost', 'B3 supervised LSTM'), ('Combined + XGBoost', 'Embedding + XGBoost'), ('Embedding + XGBoost', 'Aggregated + XGBoost'),
             ('Combined + XGBoost', 'Raw sequence'), ('B3 supervised LSTM', 'Raw sequence'), ('B3 supervised LSTM', 'Last active week (1 var.)'), ('B2 XGBoost + demographics', 'B1 RandomForest'),
             ('B1 RandomForest', 'Last active week (1 var.)'), ('Combined + XGBoost', 'Last active week (1 var.)')]:
    d = bcache[a] - bcache[b]
    res['comparisons'][f'{a} − {b}'] = {'diff': res['models'][a]['auc_pooled'] - res['models'][b]['auc_pooled'], 'ci95': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], 'share_positive': float((d > 0).mean())}
    v = res['comparisons'][f'{a} − {b}']; print(f"   {a} − {b:28s} {v['diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]", flush=True)
for k in list(store):
    if k.startswith('ref::'): del store[k]
json.dump(res, open(OUTJ, 'w')); print(f"[OK] {OUTJ}", flush=True)
