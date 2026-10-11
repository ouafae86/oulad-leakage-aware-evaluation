"""T-06 : B3' = LSTM supervisé dont l'encodeur (Masking -> LSTM(128) -> LSTM(64)) est IDENTIQUE à celui de l'autoencodeur,
suivi d'une sortie sigmoïde, entraîné de bout en bout sur les étiquettes du pli d'entraînement (conception B, GroupKFold(5)).
Usage : python eval_b3prime.py <fenêtre> [<graine> ...]        (graines des modèles supervisés ; défaut 42 43 44)
Modèles (mêmes plis que eval_windows.py) :
  B3' pretrained : encodeur initialisé avec les poids de l'autoencodeur (D-012, graine 42, sans étiquette), puis affiné
  B3' scratch    : même architecture, initialisation aléatoire
  B3 rerun       : B3 d'origine (b3.py : LSTM(64) seul, <=15 époques), rejoué ici pour une comparaison sur la même machine
  AE frozen + XGBoost / LogReg : embedding gelé du MÊME autoencodeur pré-entraîné (classifieurs à graine 42)
Protocole supervisé de B3' = celui de B3 (Adam, entropie croisée, lot 128, validation 10 % stratifiée du pli d'entraînement,
patience 3, meilleurs poids restaurés), sauf le plafond d'époques : 30 (comme l'autoencodeur) au lieu de 15.
Idempotent : chaque couple (modèle, graine) déjà présent dans results_b3prime_w<w>.json est sauté (FORCE=1 pour recalculer)."""
import sys, json, os, time, re; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from b3 import b3_fit_predict
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
def XGB(): return xgb.XGBClassifier(n_estimators=200,max_depth=5,learning_rate=0.1,eval_metric='logloss',random_state=SEED,tree_method='hist',n_jobs=1)
def LR(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED))
w = int(sys.argv[1]); seeds = [int(a) for a in sys.argv[2:]] or [42,43,44]
MAX_EP = 30
D = f'{ART}/b3prime'; os.makedirs(D, exist_ok=True)
path = f'{RES}/results_b3prime_w{w}.json'
res = json.load(open(path)) if os.path.exists(path) else {}
idx = np.where(risk_set(w))[0]; yy, gg = y[idx], gid[idx]
Xs = norm(seq[idx][:, :w, :])
folds = list(GroupKFold(5).split(Xs,yy,gg))
res['N'] = int(len(idx)); res['prev'] = float(yy.mean())

# --- 1. pré-entraînement non supervisé (même procédure que train_ae_seeds.py, graine 42) ; poids de l'encodeur conservés
PRE = f'{D}/ae_enc_w{w}_s{SEED}.npz'
if os.path.exists(PRE) and not FORCE:
    print(f'[SKIP] autoencodeur pré-entraîné existe déjà ({PRE})', flush=True)
else:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    tf.keras.utils.set_random_seed(SEED)
    ae, enc = build_ae(w, seed=SEED)
    tr, va = train_test_split(np.arange(len(Xs)), test_size=0.1, random_state=SEED, shuffle=True)
    t=time.time()
    h = ae.fit(Xs[tr], Xs[tr], validation_data=(Xs[va],Xs[va]), epochs=30, batch_size=256,
               callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)], verbose=2)
    wts = enc.get_weights()
    np.savez(PRE, idx=idx, emb=enc.predict(Xs, batch_size=512, verbose=0), n_weights=len(wts),
             loss=np.array(h.history['loss']), val_loss=np.array(h.history['val_loss']), **{f'w{i}':a for i,a in enumerate(wts)})
    print(f"pré-entraînement w={w} epochs={len(h.history['loss'])} val_loss_min={min(h.history['val_loss']):.4f} temps={time.time()-t:.0f}s", flush=True)
z = np.load(PRE); assert np.array_equal(z['idx'], idx)
enc_weights = [z[f'w{i}'] for i in range(int(z['n_weights']))]; emb = z['emb']
res['pretrain'] = {'seed':SEED, 'epochs':int(len(z['loss'])), 'val_loss_min':float(z['val_loss'].min()), 'artefact':os.path.relpath(PRE, ROOT)}

def b3prime_fit_predict(Xtr, ytr, Xte, seed, init=None):
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
    T,F = Xtr.shape[1], Xtr.shape[2]
    i = layers.Input(shape=(T,F)); x = layers.Masking()(i)
    x = layers.LSTM(128, return_sequences=True)(x); zz = layers.LSTM(64, name='emb')(x)
    enc = models.Model(i,zz); o = layers.Dense(1,activation='sigmoid')(zz)
    m = models.Model(i,o); m.compile(optimizer='adam', loss='binary_crossentropy')
    if init is not None: enc.set_weights(init)                      # tout le réseau reste entraînable (affinage de bout en bout)
    a,b = train_test_split(np.arange(len(Xtr)), test_size=0.1, random_state=seed, stratify=ytr)
    h = m.fit(Xtr[a],ytr[a], validation_data=(Xtr[b],ytr[b]), epochs=MAX_EP, batch_size=128, verbose=0,
              callbacks=[callbacks.EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)])
    return m.predict(Xte, batch_size=512, verbose=0).ravel(), len(h.history['loss'])

def run(name, seed, fit):
    """fit(tr, te) -> (scores, époques ou None). Sauvegarde les AUC par pli et les prédictions hors pli."""
    if str(seed) in res.get(name,{}) and not FORCE:
        print(f'[SKIP] {name} graine {seed} déjà évalué', flush=True); return
    a, p, ep, oof = [], [], [], np.full(len(idx), np.nan, dtype='float32'); t=time.time()
    for tr,te in folds:
        s, e = fit(tr, te); oof[te] = s
        a.append(float(roc_auc_score(yy[te],s))); p.append(float(average_precision_score(yy[te],s))); ep.append(e)
    r = {'auc':a,'ap':p}
    if ep[0] is not None: r['epochs'] = [int(e) for e in ep]
    res.setdefault(name,{})[str(seed)] = r
    np.save(f"{D}/oof_w{w}_{re.sub('[^A-Za-z0-9]+','_',name.replace(chr(39),'prime')).strip('_')}_s{seed}.npy", oof)
    json.dump(res, open(path,'w'))
    print(f"   {name:24s} graine {seed}  AUC={np.mean(a):.4f}±{np.std(a):.4f}  PR-AUC={np.mean(p):.4f}"
          + (f"  époques={r['epochs']}" if 'epochs' in r else '') + f"  ({time.time()-t:.0f}s)", flush=True)

print(f"== B3' | fenêtre {w} sem. | N={len(idx)} | taux de retrait futur={yy.mean():.3f} | graines {seeds}", flush=True)
run('AE frozen + XGBoost', SEED, lambda tr,te: (XGB().fit(emb[tr],yy[tr]).predict_proba(emb[te])[:,1], None))
run('AE frozen + LogReg',  SEED, lambda tr,te: (LR().fit(emb[tr],yy[tr]).predict_proba(emb[te])[:,1], None))
for s in seeds:
    run("B3' pretrained", s, lambda tr,te: b3prime_fit_predict(Xs[tr],yy[tr],Xs[te],s,init=enc_weights))
    run("B3' scratch",    s, lambda tr,te: b3prime_fit_predict(Xs[tr],yy[tr],Xs[te],s))
    run('B3 rerun',       s, lambda tr,te: (b3_fit_predict(Xs[tr],yy[tr],Xs[te],seed=s), None))
json.dump(res, open(path,'w'))
