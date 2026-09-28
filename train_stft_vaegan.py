"""
Train a revised conditional STFT-VAEGAN for spectrum-conditioned ground motions.

Main differences from the legacy training script:
- The generator is trained on both posterior reconstruction and prior sampling,
  so the same z ~ N(0, I) path used at inference receives adversarial and
  spectrum-consistency gradients.
- The WGAN-GP critic does not use BatchNorm, avoiding batch-coupled gradients.
- BatchNorm in the encoder/decoder is called with an explicit training flag.
- STFT amplitudes and response spectra use train-set, per-coordinate min-max
  normalization instead of one scalar min/max for the whole tensor.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import h5py
import librosa
import numpy as np
import tensorflow as tf
from scipy.ndimage import zoom
from tensorflow.keras import layers


PROJECT_DIR = Path(r"I:\sa-nga\RESS_manuscript_project")
DEFAULT_DATA_FILE = Path(r"I:\sa-nga\nga-wave\data\nga_with_spectra.h5")
DEFAULT_OUT_DIR = PROJECT_DIR / "analysis_outputs" / "stft_vaegan_revised"

N_PTS = 3000
SPEC_SHAPE = (129, 94)
N_SPECTRAL_ORDS = 32


class STFTMagnitude:
    def __init__(self, n_fft: int = 256, win_length: int = 64, hop_length: int = 32, fs: float = 50.0):
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.fs = fs
        self.window = "hann"
        self.max_freq = fs / 2.0

    def time_to_spec(self, waveform: np.ndarray) -> np.ndarray:
        waveform = np.squeeze(waveform, axis=-1)
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]

        freqs = librosa.fft_frequencies(sr=self.fs, n_fft=self.n_fft)
        keep = np.where(freqs <= self.max_freq)[0]
        out = np.zeros((waveform.shape[0], *SPEC_SHAPE), dtype=np.float32)

        for i, trace in enumerate(waveform):
            spec = librosa.stft(
                trace,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
            )
            mag = np.abs(spec)[keep, :]
            if mag.shape != SPEC_SHAPE:
                mag = zoom(mag, (SPEC_SHAPE[0] / mag.shape[0], SPEC_SHAPE[1] / mag.shape[1]), order=1)
            out[i] = mag.astype(np.float32)

        return out


def _read_record(name: str, wave_group, spe_group) -> tuple[np.ndarray, np.ndarray]:
    wave = wave_group[name][()].astype(np.float32)[:N_PTS, 0:1]
    spe = spe_group[name][()].astype(np.float32)[:, 0]
    if wave.shape[0] < N_PTS:
        wave = np.pad(wave, ((0, N_PTS - wave.shape[0]), (0, 0)), mode="constant")
    return wave, np.maximum(spe, 1e-14)


def load_dataset(data_file: Path, seed: int, train_fraction: float, max_records: int | None):
    converter = STFTMagnitude()
    with h5py.File(data_file, "r") as handle:
        wave_group = handle["data"]
        spe_group = handle["acceleration_spectra"]
        names = sorted(wave_group.keys(), key=lambda item: int(item) if item.isdigit() else item)
        if max_records is not None:
            names = names[:max_records]

        waves = np.zeros((len(names), N_PTS, 1), dtype=np.float32)
        spes = np.zeros((len(names), N_SPECTRAL_ORDS), dtype=np.float32)
        for i, name in enumerate(names):
            waves[i], spes[i] = _read_record(name, wave_group, spe_group)
            if (i + 1) % 1000 == 0:
                print(f"Loaded {i + 1}/{len(names)} records")

    specs = converter.time_to_spec(waves)
    log_specs = np.log(specs + 1e-14).astype(np.float32)
    log_spes = np.log(spes).astype(np.float32)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(names))
    train_size = int(train_fraction * len(indices))
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    spec_min = log_specs[train_idx].min(axis=0)
    spec_max = log_specs[train_idx].max(axis=0)
    spe_min = log_spes[train_idx].min(axis=0)
    spe_max = log_spes[train_idx].max(axis=0)

    spec_den = np.maximum(spec_max - spec_min, 1e-6)
    spe_den = np.maximum(spe_max - spe_min, 1e-6)

    specs_norm = np.clip((log_specs - spec_min) / spec_den, 0.0, 1.0).astype(np.float32)
    spes_norm = np.clip((log_spes - spe_min) / spe_den, 0.0, 1.0).astype(np.float32)

    norm = {
        "spec_min": spec_min.astype(np.float32),
        "spec_max": spec_max.astype(np.float32),
        "spe_min": spe_min.astype(np.float32),
        "spe_max": spe_max.astype(np.float32),
    }
    split = {"train_idx": train_idx.astype(np.int64), "test_idx": test_idx.astype(np.int64), "names": np.array(names)}
    return specs_norm, spes_norm, train_idx, test_idx, norm, split


class ConditionalEncoder(layers.Layer):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.expand = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))
        self.conv1 = layers.Conv2D(64, 5, strides=(3, 2), padding="same")
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(128, 5, strides=2, padding="same")
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(256, 5, strides=2, padding="same")
        self.bn3 = layers.BatchNormalization()
        self.conv4 = layers.Conv2D(512, 5, strides=2, padding="same")
        self.bn4 = layers.BatchNormalization()
        self.act = layers.LeakyReLU(0.2)
        self.pool = layers.GlobalAveragePooling2D()
        self.cond = layers.Dense(128, activation="relu")
        self.fuse = layers.Dense(256, activation="relu")
        self.mu = layers.Dense(latent_dim)
        self.log_var = layers.Dense(latent_dim)

    def call(self, spec, spe, training=False):
        x = self.expand(spec)
        x = self.act(self.bn1(self.conv1(x), training=training))
        x = self.act(self.bn2(self.conv2(x), training=training))
        x = self.act(self.bn3(self.conv3(x), training=training))
        x = self.act(self.bn4(self.conv4(x), training=training))
        x = self.pool(x)
        c = self.cond(spe)
        x = self.fuse(tf.concat([x, c], axis=-1))
        return self.mu(x), self.log_var(x)


class ConditionalDecoder(layers.Layer):
    def __init__(self):
        super().__init__()
        self.cond = layers.Dense(128, activation="relu")
        self.fc = layers.Dense(6 * 6 * 512, activation="relu")
        self.reshape = layers.Reshape((6, 6, 512))
        self.deconv1 = layers.Conv2DTranspose(256, 5, strides=2, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.deconv2 = layers.Conv2DTranspose(128, 5, strides=2, padding="same")
        self.bn2 = layers.BatchNormalization()
        self.deconv3 = layers.Conv2DTranspose(64, 5, strides=2, padding="same")
        self.bn3 = layers.BatchNormalization()
        self.deconv4 = layers.Conv2DTranspose(1, 5, strides=(3, 2), padding="same", activation="sigmoid")
        self.crop = layers.Cropping2D(cropping=((7, 8), (1, 1)))
        self.squeeze = layers.Lambda(lambda x: tf.squeeze(x, axis=-1))
        self.act = layers.LeakyReLU(0.2)

    def call(self, z, spe, training=False):
        c = self.cond(spe)
        x = self.fc(tf.concat([z, c], axis=-1))
        x = self.reshape(x)
        x = self.act(self.bn1(self.deconv1(x), training=training))
        x = self.act(self.bn2(self.deconv2(x), training=training))
        x = self.act(self.bn3(self.deconv3(x), training=training))
        return self.squeeze(self.crop(self.deconv4(x)))


class Generator(tf.keras.Model):
    def __init__(self, latent_dim: int = 100):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = ConditionalEncoder(latent_dim)
        self.decoder = ConditionalDecoder()

    def sample(self, mu, log_var):
        eps = tf.random.normal(tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * eps

    def call(self, spec, spe, training=False):
        mu, log_var = self.encoder(spec, spe, training=training)
        z = self.sample(mu, log_var)
        recon = self.decoder(z, spe, training=training)
        return recon, mu, log_var

    def generate(self, spe, training=False):
        z = tf.random.normal((tf.shape(spe)[0], self.latent_dim))
        return self.decoder(z, spe, training=training)


class Critic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.expand = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))
        self.conv1 = layers.Conv2D(64, 5, strides=(3, 2), padding="same")
        self.conv2 = layers.Conv2D(128, 5, strides=2, padding="same")
        self.conv3 = layers.Conv2D(256, 5, strides=2, padding="same")
        self.conv4 = layers.Conv2D(512, 5, strides=2, padding="same")
        self.pool = layers.GlobalAveragePooling2D()
        self.cond = layers.Dense(128, activation="relu")
        self.fuse = layers.Dense(128, activation="relu")
        self.out = layers.Dense(1)
        self.act = layers.LeakyReLU(0.2)

    def call(self, spec, spe, training=False):
        del training
        x = self.expand(spec)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.pool(x)
        c = self.cond(spe)
        return self.out(self.fuse(tf.concat([x, c], axis=-1)))


class SpectralRegressor(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.expand = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))
        self.conv1 = layers.Conv2D(64, 5, strides=(3, 2), padding="same")
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(128, 5, strides=2, padding="same")
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(256, 5, strides=2, padding="same")
        self.bn3 = layers.BatchNormalization()
        self.conv4 = layers.Conv2D(512, 5, strides=2, padding="same")
        self.bn4 = layers.BatchNormalization()
        self.pool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(128, activation="relu")
        self.out = layers.Dense(N_SPECTRAL_ORDS, activation="sigmoid")
        self.act = layers.LeakyReLU(0.2)

    def call(self, spec, training=False):
        x = self.expand(spec)
        x = self.act(self.bn1(self.conv1(x), training=training))
        x = self.act(self.bn2(self.conv2(x), training=training))
        x = self.act(self.bn3(self.conv3(x), training=training))
        x = self.act(self.bn4(self.conv4(x), training=training))
        return self.out(self.fc(self.pool(x)))


def gradient_penalty(critic: Critic, real_spec, fake_spec, spe):
    batch = tf.shape(real_spec)[0]
    alpha = tf.random.uniform((batch, 1, 1), 0.0, 1.0)
    mixed = alpha * real_spec + (1.0 - alpha) * fake_spec
    with tf.GradientTape() as tape:
        tape.watch(mixed)
        score = critic(mixed, spe, training=True)
    grad = tape.gradient(score, mixed)
    grad = tf.reshape(grad, (batch, -1))
    return tf.reduce_mean((tf.norm(grad, axis=1) - 1.0) ** 2)


def make_dataset(specs, spes, indices, batch_size, shuffle, seed):
    ds = tf.data.Dataset.from_tensor_slices((specs[indices], spes[indices]))
    if shuffle:
        ds = ds.shuffle(len(indices), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size, drop_remainder=shuffle).prefetch(tf.data.AUTOTUNE)


def train(args):
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    specs, spes, train_idx, test_idx, norm, split = load_dataset(
        args.data_file,
        args.seed,
        args.train_fraction,
        args.max_records,
    )
    np.savez_compressed(args.out_dir / "normalization_and_split.npz", **norm, **split)

    train_ds = make_dataset(specs, spes, train_idx, args.batch_size, True, args.seed)
    test_ds = make_dataset(specs, spes, test_idx, args.batch_size, False, args.seed)

    generator = Generator(args.latent_dim)
    critic = Critic()
    regressor = SpectralRegressor()

    # Build variables before checkpointing.
    dummy_spec = tf.zeros((1, *SPEC_SHAPE), dtype=tf.float32)
    dummy_spe = tf.zeros((1, N_SPECTRAL_ORDS), dtype=tf.float32)
    _ = generator(dummy_spec, dummy_spe, training=False)
    _ = critic(dummy_spec, dummy_spe, training=False)
    _ = regressor(dummy_spec, training=False)

    opt_g = tf.keras.optimizers.Adam(args.lr_g, beta_1=0.0, beta_2=0.9)
    opt_d = tf.keras.optimizers.Adam(args.lr_d, beta_1=0.0, beta_2=0.9)
    opt_r = tf.keras.optimizers.Adam(args.lr_r)

    ckpt = tf.train.Checkpoint(generator=generator, critic=critic, regressor=regressor, opt_g=opt_g, opt_d=opt_d, opt_r=opt_r)
    manager = tf.train.CheckpointManager(ckpt, str(args.out_dir / "checkpoints"), max_to_keep=args.max_checkpoints)

    @tf.function
    def train_regressor(real_spec, spe):
        with tf.GradientTape() as tape:
            pred = regressor(real_spec, training=True)
            loss = tf.reduce_mean(tf.square(pred - spe))
        grads = tape.gradient(loss, regressor.trainable_variables)
        opt_r.apply_gradients(zip(grads, regressor.trainable_variables))
        return loss

    @tf.function
    def train_critic(real_spec, spe):
        with tf.GradientTape() as tape:
            fake = generator.generate(spe, training=True)
            real_score = critic(real_spec, spe, training=True)
            fake_score = critic(fake, spe, training=True)
            gp = gradient_penalty(critic, real_spec, fake, spe)
            loss = tf.reduce_mean(fake_score) - tf.reduce_mean(real_score) + args.lambda_gp * gp
        grads = tape.gradient(loss, critic.trainable_variables)
        opt_d.apply_gradients(zip(grads, critic.trainable_variables))
        return loss, gp, tf.reduce_mean(real_score), tf.reduce_mean(fake_score)

    @tf.function
    def train_generator(real_spec, spe):
        with tf.GradientTape() as tape:
            recon, mu, log_var = generator(real_spec, spe, training=True)
            fake = generator.generate(spe, training=True)

            recon_loss = tf.reduce_mean(tf.square(recon - real_spec))
            kl_loss = -0.5 * tf.reduce_mean(1.0 + log_var - tf.square(mu) - tf.exp(log_var))
            adv_loss = -tf.reduce_mean(critic(fake, spe, training=True))
            reg_prior = tf.reduce_mean(tf.square(regressor(fake, training=False) - spe))
            reg_recon = tf.reduce_mean(tf.square(regressor(recon, training=False) - spe))
            reg_loss = reg_prior + 0.5 * reg_recon
            loss = (
                args.lambda_recon * recon_loss
                + args.lambda_kl * kl_loss
                + args.lambda_adv * adv_loss
                + args.lambda_reg * reg_loss
            )
        grads = tape.gradient(loss, generator.trainable_variables)
        opt_g.apply_gradients(zip(grads, generator.trainable_variables))
        return loss, recon_loss, kl_loss, adv_loss, reg_prior, reg_recon

    @tf.function
    def eval_batch(real_spec, spe):
        recon, _, _ = generator(real_spec, spe, training=False)
        fake = generator.generate(spe, training=False)
        recon_mse = tf.reduce_mean(tf.square(recon - real_spec))
        prior_reg = tf.reduce_mean(tf.square(regressor(fake, training=False) - spe))
        recon_reg = tf.reduce_mean(tf.square(regressor(recon, training=False) - spe))
        return recon_mse, prior_reg, recon_reg

    history_path = args.out_dir / "training_history.csv"
    history_path.write_text(
        "epoch,g_loss,d_loss,r_loss,recon,kl,adv,reg_prior,reg_recon,gp,d_real,d_fake,val_recon,val_reg_prior,val_reg_recon,seconds\n",
        encoding="utf-8",
    )

    config = vars(args).copy()
    config["data_file"] = str(args.data_file)
    config["out_dir"] = str(args.out_dir)
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        sums = {key: 0.0 for key in ["g", "d", "r", "recon", "kl", "adv", "reg_prior", "reg_recon", "gp", "d_real", "d_fake"]}
        n_batches = 0
        d_steps = 0

        for real_spec, spe in train_ds:
            r_loss = train_regressor(real_spec, spe)
            for _ in range(args.n_critic):
                d_loss, gp, d_real, d_fake = train_critic(real_spec, spe)
                sums["d"] += float(d_loss)
                sums["gp"] += float(gp)
                sums["d_real"] += float(d_real)
                sums["d_fake"] += float(d_fake)
                d_steps += 1

            g_loss, recon, kl, adv, reg_prior, reg_recon = train_generator(real_spec, spe)
            sums["g"] += float(g_loss)
            sums["r"] += float(r_loss)
            sums["recon"] += float(recon)
            sums["kl"] += float(kl)
            sums["adv"] += float(adv)
            sums["reg_prior"] += float(reg_prior)
            sums["reg_recon"] += float(reg_recon)
            n_batches += 1

        val_recon = []
        val_reg_prior = []
        val_reg_recon = []
        for real_spec, spe in test_ds:
            a, b, c = eval_batch(real_spec, spe)
            val_recon.append(float(a))
            val_reg_prior.append(float(b))
            val_reg_recon.append(float(c))

        seconds = time.time() - start
        row = {
            "epoch": epoch,
            "g_loss": sums["g"] / n_batches,
            "d_loss": sums["d"] / d_steps,
            "r_loss": sums["r"] / n_batches,
            "recon": sums["recon"] / n_batches,
            "kl": sums["kl"] / n_batches,
            "adv": sums["adv"] / n_batches,
            "reg_prior": sums["reg_prior"] / n_batches,
            "reg_recon": sums["reg_recon"] / n_batches,
            "gp": sums["gp"] / d_steps,
            "d_real": sums["d_real"] / d_steps,
            "d_fake": sums["d_fake"] / d_steps,
            "val_recon": float(np.mean(val_recon)),
            "val_reg_prior": float(np.mean(val_reg_prior)),
            "val_reg_recon": float(np.mean(val_reg_recon)),
            "seconds": seconds,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(",".join(str(row[key]) for key in row) + "\n")

        print(
            f"Epoch {epoch:04d}/{args.epochs} "
            f"G={row['g_loss']:.4f} D={row['d_loss']:.4f} R={row['r_loss']:.4f} "
            f"recon={row['recon']:.4f} priorReg={row['reg_prior']:.4f} "
            f"valRecon={row['val_recon']:.4f} valPriorReg={row['val_reg_prior']:.4f} "
            f"time={seconds:.1f}s"
        )

        if epoch % args.save_every == 0 or epoch == args.epochs:
            saved = manager.save(checkpoint_number=epoch)
            print(f"Saved checkpoint: {saved}")

    final_ckpt = manager.save(checkpoint_number=args.epochs)
    generator.save_weights(str(args.out_dir / "generator_final_weights.h5"))
    regressor.save_weights(str(args.out_dir / "regressor_final_weights.h5"))
    critic.save_weights(str(args.out_dir / "critic_final_weights.h5"))
    print(f"Saved final checkpoint: {final_ckpt}")
    print(f"Training complete. Outputs: {args.out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train revised conditional STFT-VAEGAN")
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-critic", type=int, default=5)
    parser.add_argument("--lr-g", type=float, default=5e-5)
    parser.add_argument("--lr-d", type=float, default=5e-5)
    parser.add_argument("--lr-r", type=float, default=8e-5)
    parser.add_argument("--lambda-recon", type=float, default=3.0)
    parser.add_argument("--lambda-kl", type=float, default=0.1)
    parser.add_argument("--lambda-adv", type=float, default=0.05)
    parser.add_argument("--lambda-reg", type=float, default=1.0)
    parser.add_argument("--lambda-gp", type=float, default=10.0)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--max-checkpoints", type=int, default=5)
    parser.add_argument("--max-records", type=int, default=None, help="Optional small-data smoke-test limit")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
