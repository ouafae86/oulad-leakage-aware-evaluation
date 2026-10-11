import sys, time, json, numpy as np; import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.model_selection import train_test_split
w = int(sys.argv[1]); max_ep = int(sys.argv[2])
skip_if_exists(f'{ART}/ae_w{w}.npz', f'autoencodeur w={w}')
m = risk_set(w); idx = np.where(m)[0]
X = norm(seq[idx][:, :w, :])
ae, enc = build_ae(w)
emb_untrained = enc.predict(X, batch_size=512, verbose=0)           # contrôle C4 : encodeur LSTM non entraîné
tr, va = train_test_split(np.arange(len(X)), test_size=0.1, random_state=SEED, shuffle=True)  # validation MÉLANGÉE
from tensorflow.keras.callbacks import EarlyStopping
t=time.time()
h = ae.fit(X[tr], X[tr], validation_data=(X[va],X[va]), epochs=max_ep, batch_size=256,
           callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)], verbose=2)
emb = enc.predict(X, batch_size=512, verbose=0)
np.savez(f'{ART}/ae_w{w}.npz', idx=idx, emb=emb, emb_untrained=emb_untrained,
         loss=np.array(h.history['loss']), val_loss=np.array(h.history['val_loss']))
print(f"w={w} N={len(idx)} epochs={len(h.history['loss'])} val_loss={h.history['val_loss'][-1]:.4f} temps={time.time()-t:.0f}s", flush=True)
