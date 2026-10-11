"""
lstm_autoencoder.py
─────────────────────────────────────────────────────────────────
Section 4.6 de la methodologie — Autoencodeur LSTM non supervise.

Contribution centrale de l'article : apprendre une representation
compacte des sequences comportementales SANS utiliser les labels
de decrochage, via une tache de reconstruction. L'encodeur produit
un embedding de dimension latent_dim par etudiant, utilise ensuite
en entree d'un classifieur simple (evaluation.py).

Usage :
    python3 lstm_autoencoder.py [--epochs N] [--latent-dim N] [--quick]
─────────────────────────────────────────────────────────────────
"""

import os
import argparse
import numpy as np
import pandas as pd

INTERIM_DIR = os.path.join(os.path.dirname(__file__), "..", "interim")
RANDOM_SEED = 42


def build_autoencoder(n_timesteps: int, n_features: int, latent_dim: int):
    """
    Architecture : Encoder LSTM (sequence -> vecteur latent) puis
    Decoder LSTM (vecteur latent -> sequence reconstruite), entraines
    conjointement sur l'erreur de reconstruction (MSE).
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    tf.random.set_seed(RANDOM_SEED)

    inputs = layers.Input(shape=(n_timesteps, n_features), name="clickstream")
    x = layers.Masking()(inputs)
    x = layers.LSTM(128, return_sequences=True)(x)
    latent = layers.LSTM(latent_dim, name="embedding")(x)

    x = layers.RepeatVector(n_timesteps)(latent)
    x = layers.LSTM(128, return_sequences=True)(x)
    outputs = layers.TimeDistributed(layers.Dense(n_features), name="reconstruction")(x)

    autoencoder = models.Model(inputs, outputs, name="lstm_autoencoder")
    encoder = models.Model(inputs, latent, name="lstm_encoder")

    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder, encoder


def normalize_sequences(sequences: np.ndarray):
    """
    Normalisation log1p (les clics sont tres asymetriques : beaucoup de
    zeros, quelques pics tres eleves) puis standardisation par feature.
    Les parametres de normalisation sont retournes pour etre reappliques
    a l'identique sur toute nouvelle sequence (pas de fuite : ce sont des
    statistiques globales sur les FEATURES, pas sur la cible).
    """
    log_seq = np.log1p(sequences)
    mean = log_seq.mean(axis=(0, 1), keepdims=True)
    std = log_seq.std(axis=(0, 1), keepdims=True) + 1e-6
    normalized = (log_seq - mean) / std
    return normalized.astype("float32"), mean, std


def train_and_extract_embeddings(sequences: np.ndarray, latent_dim: int = 64,
                                  epochs: int = 30, batch_size: int = 128,
                                  validation_split: float = 0.1, verbose: int = 2):
    n_students, n_timesteps, n_features = sequences.shape
    normalized, mean, std = normalize_sequences(sequences)

    autoencoder, encoder = build_autoencoder(n_timesteps, n_features, latent_dim)

    from tensorflow.keras.callbacks import EarlyStopping
    history = autoencoder.fit(
        normalized, normalized,
        epochs=epochs, batch_size=batch_size, validation_split=validation_split,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
        verbose=verbose,
    )

    embeddings = encoder.predict(normalized, batch_size=batch_size, verbose=0)
    reconstruction = autoencoder.predict(normalized, batch_size=batch_size, verbose=0)
    recon_mse_per_student = ((reconstruction - normalized) ** 2).mean(axis=(1, 2))

    return embeddings, history.history, recon_mse_per_student


def validation_report(embeddings, history, recon_mse, cohort):
    print("=" * 70)
    print("RAPPORT DE VALIDATION — lstm_autoencoder.py")
    print("=" * 70)
    print(f"\nEmbeddings : shape={embeddings.shape}")
    print(f"Perte finale (train) : {history['loss'][-1]:.4f}")
    print(f"Perte finale (val)   : {history['val_loss'][-1]:.4f}")
    print(f"Nb epochs effectuees  : {len(history['loss'])}")

    print(f"\nErreur de reconstruction par etudiant :")
    print(f"  moyenne={recon_mse.mean():.4f}  std={recon_mse.std():.4f}  "
          f"min={recon_mse.min():.4f}  max={recon_mse.max():.4f}")

    print(f"\nErreur de reconstruction par classe (verification qualitative) :")
    df = cohort[["dropout"]].copy()
    df["recon_mse"] = recon_mse
    print(df.groupby("dropout")["recon_mse"].mean().to_string(float_format=lambda x: f"{x:.4f}"))
    print("(Non attendu de grand ecart : l'autoencodeur reconstruit la forme du signal,")
    print(" pas le label — un ecart present serait plutot le signe d'une fuite a inspecter.)")

    print(f"\nStatistiques des embeddings (verifier l'absence de colonnes mortes) :")
    emb_std = embeddings.std(axis=0)
    print(f"  dimensions avec std < 1e-3 (potentiellement mortes) : {(emb_std < 1e-3).sum()} / {embeddings.shape[1]}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--quick", action="store_true", help="Run rapide de validation (peu d'epochs)")
    args = parser.parse_args()
    if args.quick:
        args.epochs = 3

    cohort = pd.read_parquet(f"{INTERIM_DIR}/cohort.parquet")
    sequences = np.load(f"{INTERIM_DIR}/sequences.npy")
    assert sequences.shape[0] == len(cohort), "Desalignement cohort/sequences !"

    embeddings, history, recon_mse = train_and_extract_embeddings(
        sequences, latent_dim=args.latent_dim, epochs=args.epochs
    )
    validation_report(embeddings, history, recon_mse, cohort)

    np.save(f"{INTERIM_DIR}/embeddings.npy", embeddings)
    np.save(f"{INTERIM_DIR}/recon_mse.npy", recon_mse)
    import json
    with open(f"{INTERIM_DIR}/autoencoder_history.json", "w") as f:
        json.dump(history, f)
    print(f"\n[OK] Embeddings ({embeddings.shape}) sauvegardes -> {INTERIM_DIR}/embeddings.npy")
    if args.quick:
        print("[ATTENTION] --quick actif : ces embeddings sont sous-entraines (3 epochs), "
              "usage validation du pipeline uniquement, PAS pour les resultats finaux de l'article.")
