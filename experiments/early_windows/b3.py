import numpy as np
from common import *
def b3_fit_predict(Xtr, ytr, Xte, seed=SEED, max_ep=15):
    """B3 : LSTM supervisé (Masking -> LSTM(64) -> sigmoïde), identique à l'article ; validation 10 % mélangée, patience 3."""
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    from sklearn.model_selection import train_test_split
    tf.keras.backend.clear_session(); tf.random.set_seed(seed)
    T,F = Xtr.shape[1], Xtr.shape[2]
    i = layers.Input(shape=(T,F)); x = layers.Masking()(i); x = layers.LSTM(64)(x); o = layers.Dense(1,activation='sigmoid')(x)
    m = models.Model(i,o); m.compile(optimizer='adam', loss='binary_crossentropy')
    if len(Xtr) >= 50 and ytr.min()!=ytr.max():
        a,b = train_test_split(np.arange(len(Xtr)), test_size=0.1, random_state=seed, stratify=ytr)
    else:
        a,b = np.arange(len(Xtr)), np.arange(len(Xtr))
    m.fit(Xtr[a],ytr[a], validation_data=(Xtr[b],ytr[b]), epochs=max_ep, batch_size=64 if len(Xtr)<3000 else 128, verbose=0,
          callbacks=[callbacks.EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)])
    return m.predict(Xte, batch_size=512, verbose=0).ravel()
