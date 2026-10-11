"""T-19 : le gain du pré-entraînement de B3' dépend-il de la graine de l'autoencodeur ?
Usage : python eval_b3prime_seeds.py <fenêtre> [<graine d'autoencodeur> ...]     (défaut 42 43 44 45 46)
Pour chaque graine d'autoencodeur a : (1) autoencodeur entraîné sans étiquette comme dans eval_b3prime.py (D-012, D-021),
poids de l'encodeur conservés ; (2) B3' initialisé avec ces poids puis affiné de bout en bout, avec les graines
supervisées 42, 43, 44, sur les plis enregistrés de l'analyse principale. La graine a = 42 est celle de T-06 :
ses prédictions sont RÉUTILISÉES, pas recalculées. B3' entraîné de zéro (T-06) ne dépend pas de l'autoencodeur.
Résultat : par graine d'autoencodeur, AUC moyenne des plis et gain sur B3' de zéro, avec bootstrap apparié par étudiant
(statistique « moyenne des plis », D-031) -> results/early_windows/results_b3prime_seeds_w12.json.
Idempotent : poids et prédictions existants sont réutilisés (FORCE=1 pour recalculer les graines autres que 42)."""
import sys, json, os, time, glob; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from boot import Boot
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
w = int(sys.argv[1]); ae_seeds = [int(a) for a in sys.argv[2:]] or [42, 43, 44, 45, 46]
SUP = [42, 43, 44]; MAX_EP = 30; NB = 2000
D = f'{ART}/b3prime'; OUTJ = f'{RES}/results_b3prime_seeds_w{w}.json'
skip_if_exists(OUTJ, f"B3' : graines d'autoencodeur w={w}")
z = np.load(f'{ART}/oof/oof_B_w{w}.npz'); idx, fold = z['idx'], z['fold']; yy = y[idx]
assert np.array_equal(idx, np.where(risk_set(w))[0])
Xs = norm(seq[idx][:, :w, :])

def pretrain(a):
    """Autoencodeur de pré-entraînement (même code que eval_b3prime.py), graine a ; renvoie les poids de l'encodeur."""
    f = f'{D}/ae_enc_w{w}_s{a}.npz'
    if not os.path.exists(f) or (FORCE and a != SEED):
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping
        tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(a)
        ae, enc = build_ae(w, seed=a)
        tr, va = train_test_split(np.arange(len(Xs)), test_size=0.1, random_state=a, shuffle=True)
        t = time.time()
        h = ae.fit(Xs[tr], Xs[tr], validation_data=(Xs[va], Xs[va]), epochs=30, batch_size=256,
                   callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)], verbose=0)
        wts = enc.get_weights()
        np.savez(f, idx=idx, emb=enc.predict(Xs, batch_size=512, verbose=0), n_weights=len(wts),
                 loss=np.array(h.history['loss']), val_loss=np.array(h.history['val_loss']), **{f'w{i}': v for i, v in enumerate(wts)})
        print(f"pré-entraînement graine {a} : epochs={len(h.history['loss'])} val_loss_min={min(h.history['val_loss']):.4f} temps={time.time()-t:.0f}s", flush=True)
    else: print(f'[SKIP] autoencodeur de pré-entraînement graine {a} existe déjà', flush=True)
    q = np.load(f); assert np.array_equal(q['idx'], idx)
    return [q[f'w{i}'] for i in range(int(q['n_weights']))], {'epochs': int(len(q['loss'])), 'val_loss_min': float(q['val_loss'].min())}

def fit_predict(Xtr, ytr, Xte, seed, init):
    """B3' pré-entraîné (même code que eval_b3prime.py)."""
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
    i = layers.Input(shape=Xtr.shape[1:]); x = layers.Masking()(i)
    x = layers.LSTM(128, return_sequences=True)(x); zz = layers.LSTM(64, name='emb')(x)
    enc = models.Model(i, zz); m = models.Model(i, layers.Dense(1, activation='sigmoid')(zz)); m.compile(optimizer='adam', loss='binary_crossentropy')
    enc.set_weights(init)
    a, b = train_test_split(np.arange(len(Xtr)), test_size=0.1, random_state=seed, stratify=ytr)
    h = m.fit(Xtr[a], ytr[a], validation_data=(Xtr[b], ytr[b]), epochs=MAX_EP, batch_size=128, verbose=0,
              callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])
    return m.predict(Xte, batch_size=512, verbose=0).ravel(), len(h.history['loss'])

fa = lambda s: [float(roc_auc_score(yy[fold == k], s[fold == k])) for k in range(5)]
scratch = [np.load(f) for f in sorted(glob.glob(f'{D}/oof_w{w}_B3prime_scratch_s*.npy'))]; assert len(scratch) == len(SUP)
res = {'meta': {'window': w, 'ae_seeds': ae_seeds, 'supervised_seeds': SUP, 'n_boot': NB, 'statistic': "AUC-ROC par pli puis moyenne des plis, moyenne des graines supervisées",
                'note': "graine d'autoencodeur 42 = résultats de T-06 réutilisés ; B3' de zéro = T-06"}, 'ae': {}}
bt = Boot(gid[idx], yy, NB, seed=SEED); b_scr = bt.model_folds(scratch, fold); m_scr = float(np.mean([fa(s) for s in scratch]))
res['scratch'] = {'auc': m_scr, 'fold_auc': [float(v) for v in np.mean([fa(s) for s in scratch], axis=0)]}
allpre = []; t0 = time.time()
for a in ae_seeds:
    wts, info = pretrain(a); runs = []; eps = {}
    for sd in SUP:
        f = f'{D}/oof_w{w}_B3prime_pretrained_s{sd}.npy' if a == SEED else f'{D}/oof_w{w}_B3prime_pretrained_ae{a}_s{sd}.npy'
        if os.path.exists(f) and not (FORCE and a != SEED): s = np.load(f)
        else:
            s = np.full(len(idx), np.nan, dtype='float32'); ep = []
            for k in range(5):
                tr, te = np.where(fold != k)[0], np.where(fold == k)[0]; s[te], e = fit_predict(Xs[tr], yy[tr], Xs[te], sd, wts); ep.append(int(e))
            np.save(f, s); eps[str(sd)] = ep
        runs.append(s)
    allpre += runs; b = bt.model_folds(runs, fold); d = b - b_scr; mp = float(np.mean([fa(s) for s in runs]))
    res['ae'][str(a)] = {**info, 'auc': mp, 'auc_by_supervised_seed': [float(np.mean(fa(s))) for s in runs], 'epochs': eps,
                         'gain_vs_scratch': {'diff': mp - m_scr, 'ci95': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], 'share_positive': float((d > 0).mean())}}
    v = res['ae'][str(a)]['gain_vs_scratch']
    print(f"graine d'autoencodeur {a} : B3' pré-entraîné {mp:.4f} | de zéro {m_scr:.4f} | gain {v['diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(OUTJ + '.partiel', 'w'))
b = bt.model_folds(allpre, fold); d = b - b_scr; mp = float(np.mean([fa(s) for s in allpre]))
res['all_ae_seeds'] = {'auc': mp, 'n_runs': len(allpre), 'gain_vs_scratch': {'diff': mp - m_scr, 'ci95': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], 'share_positive': float((d > 0).mean())}}
raw = z[f"m{list(z['names']).index('Raw sequence')}"]; dr = b - bt.auc_folds(raw, fold); mr = float(np.mean(fa(raw)))
res['all_ae_seeds']['vs_raw_sequence'] = {'diff': mp - mr, 'ci95': [float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5))]}
v = res['all_ae_seeds']['gain_vs_scratch']; print(f"toutes graines d'autoencodeur : gain {v['diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] ; vs séquence brute {mp-mr:+.4f}", flush=True)
json.dump(res, open(OUTJ, 'w')); os.remove(OUTJ + '.partiel'); print(f"[OK] {OUTJ}", flush=True)
