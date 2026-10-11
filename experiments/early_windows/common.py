import os, sys, numpy as np, pandas as pd
from pathlib import Path
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
# Racine du projet : variable PROJECT_ROOT, sinon deux niveaux au-dessus de ce fichier (experiments/early_windows/common.py)
ROOT = Path(os.environ.get('PROJECT_ROOT', Path(__file__).resolve().parents[2]))
UP  = str(ROOT/'interim')                       # ENTREES (produites par src/ : cohort.parquet, sequences.npy, ...)
ART = str(ROOT/'interim'/'early_windows')       # ARTEFACTS lourds (ae_w*.npz, agg_w*.parquet) - non versionnés
RES = str(ROOT/'results'/'early_windows')       # RESULTATS légers (results_*.json) - versionnés
OUT = ART                                       # alias de compatibilité
os.makedirs(ART, exist_ok=True); os.makedirs(RES, exist_ok=True)
SEED=42
FORCE = os.environ.get('FORCE','0')=='1'
def skip_if_exists(path, label=''):
    """Garde-fou : ne JAMAIS recalculer/écraser un résultat existant, sauf FORCE=1."""
    if os.path.exists(path) and not FORCE:
        print(f"[SKIP] {label or os.path.basename(path)} existe déjà ({path}) — FORCE=1 pour recalculer", flush=True)
        sys.exit(0)
cohort = pd.read_parquet(f'{UP}/cohort.parquet')
seq = np.load(f'{UP}/sequences.npy')            # (N,39,20)
y = cohort['dropout'].values.astype(int)
gid = cohort['id_student'].values
unreg = cohort['date_unregistration'].astype('float').values   # NaN si inconnu
def risk_set(w):
    """Étudiants encore inscrits à la fin de la fenêtre (jour 7*w) : on ne prédit pas un retrait déjà observé."""
    return ~(unreg < 7*w)          # NaN -> comparaison False -> reste dans l'ensemble
def norm(x):
    l = np.log1p(x); m = l.mean(axis=(0,1),keepdims=True); s = l.std(axis=(0,1),keepdims=True)+1e-6
    return ((l-m)/s).astype('float32')
def build_ae(T, F=20, latent=64, seed=SEED):
    import tensorflow as tf
    from tensorflow.keras import layers, models
    tf.random.set_seed(seed)
    i = layers.Input(shape=(T,F)); x = layers.Masking()(i)
    x = layers.LSTM(128, return_sequences=True)(x); z = layers.LSTM(latent, name='emb')(x)
    x = layers.RepeatVector(T)(z); x = layers.LSTM(128, return_sequences=True)(x)
    o = layers.TimeDistributed(layers.Dense(F))(x)
    ae = models.Model(i,o); enc = models.Model(i,z); ae.compile(optimizer='adam', loss='mse')
    return ae, enc
