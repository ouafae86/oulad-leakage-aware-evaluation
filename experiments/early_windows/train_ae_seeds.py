"""T-05 : même autoencodeur que train_ae.py (D-012), ré-entraîné avec une graine donnée.
Usage : python train_ae_seeds.py <fenêtre> <époques max> <graine>
La graine fixe l'initialisation des poids, l'ordre des lots et le découpage de validation (10 % mélangés).
Sortie : interim/early_windows/seeds/ae_w<w>_s<graine>.npz (ae_w<w>.npz, artefact d'origine, n'est jamais touché)."""
import sys, time, numpy as np; import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.model_selection import train_test_split
w = int(sys.argv[1]); max_ep = int(sys.argv[2]); seed = int(sys.argv[3])
os.makedirs(f'{ART}/seeds', exist_ok=True)
skip_if_exists(f'{ART}/seeds/ae_w{w}_s{seed}.npz', f'autoencodeur w={w} graine={seed}')
import tensorflow as tf
tf.keras.utils.set_random_seed(seed)                                # python, numpy et tensorflow
m = risk_set(w); idx = np.where(m)[0]
X = norm(seq[idx][:, :w, :])
ae, enc = build_ae(w, seed=seed)
emb_untrained = enc.predict(X, batch_size=512, verbose=0)           # contrôle C4 : encodeur LSTM non entraîné (même graine)
tr, va = train_test_split(np.arange(len(X)), test_size=0.1, random_state=seed, shuffle=True)  # validation MÉLANGÉE
from tensorflow.keras.callbacks import EarlyStopping
t=time.time()
h = ae.fit(X[tr], X[tr], validation_data=(X[va],X[va]), epochs=max_ep, batch_size=256,
           callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)], verbose=2)
emb = enc.predict(X, batch_size=512, verbose=0)
np.savez(f'{ART}/seeds/ae_w{w}_s{seed}.npz', idx=idx, emb=emb, emb_untrained=emb_untrained, seed=seed,
         loss=np.array(h.history['loss']), val_loss=np.array(h.history['val_loss']))
print(f"w={w} graine={seed} N={len(idx)} epochs={len(h.history['loss'])} val_loss_min={min(h.history['val_loss']):.4f} temps={time.time()-t:.0f}s", flush=True)
