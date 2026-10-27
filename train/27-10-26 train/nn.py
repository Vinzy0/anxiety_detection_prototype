import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from common import *

EPOCHS, PATIENCE, BATCH = 120, 12, 128


class Prep:
    def __init__(self, D, tr, kind):
        self.D, self.kind = D, kind
        self.cols = np.arange(D["X"].shape[1]) if kind == "mlp" else np.array(D["extra_idx"] + D["ind_idx"])
        X = D["X"][tr][:, self.cols]
        self.mu = np.nan_to_num(np.nanmean(X, 0))
        sd = np.nanstd(X, 0)
        self.sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
        if kind == "cnn":
            S = self.demean(D["seq"][np.sort(tr)])
            self.cm = np.nan_to_num(np.nanmean(S, (0, 2)))
            cs = np.nanstd(S, (0, 2))
            self.cs = np.where(np.isfinite(cs) & (cs > 1e-6), cs, 1.0)

    @staticmethod
    def demean(S):
        S = S.astype(np.float32)
        return S - np.nanmean(S, axis=2, keepdims=True)

    def __call__(self, idx):
        X = self.D["X"][idx]
        ex = np.hstack([np.clip(np.nan_to_num((X[:, self.cols] - self.mu) / self.sd), -10, 10),
                        np.isnan(X[:, self.D["ind_idx"]])]).astype(np.float32)
        if self.kind == "mlp":
            return ex
        S = self.demean(self.D["seq"][idx])
        S = np.clip(np.nan_to_num((S - self.cm[None, :, None]) / self.cs[None, :, None]), -10, 10)
        return [S.transpose(0, 2, 1).astype(np.float32), ex]

    def state(self):
        s = dict(mu=self.mu, sd=self.sd, cols=self.cols)
        return {**s, **(dict(cm=self.cm, cs=self.cs) if self.kind == "cnn" else {})}


def build(kind, shapes, k=2):
    reg = tf.keras.regularizers.L2(1e-4)
    L = tf.keras.layers
    if kind == "mlp":
        i = tf.keras.Input(shape=(shapes[0],))
        x = L.Dropout(0.2)(i)
        x = L.BatchNormalization()(L.Dense(256, activation="relu", kernel_regularizer=reg)(x))
        x = L.Dropout(0.4)(x)
        x = L.Dropout(0.3)(L.Dense(64, activation="relu", kernel_regularizer=reg)(x))
        return tf.keras.Model(i, L.Dense(k, activation="softmax")(x))
    s, e = tf.keras.Input(shape=shapes[0]), tf.keras.Input(shape=(shapes[1],))
    x = L.GaussianNoise(0.05)(s)
    for f in (48, 64, 96):
        x = L.Conv1D(f, 5, padding="same", kernel_regularizer=reg)(x)
        x = L.MaxPool1D(2)(L.SpatialDropout1D(0.2)(L.ReLU()(L.BatchNormalization()(x))))
    x = L.Concatenate()([L.GlobalAveragePooling1D()(x), L.GlobalMaxPooling1D()(x),
                         L.Dense(64, activation="relu", kernel_regularizer=reg)(L.Dropout(0.2)(e))])
    x = L.Dropout(0.3)(L.Dense(64, activation="relu", kernel_regularizer=reg)(L.Dropout(0.4)(x)))
    return tf.keras.Model([s, e], L.Dense(k, activation="softmax")(x))


def make_model(kind, inputs):
    return build(kind, [inputs.shape[1]] if kind == "mlp" else [inputs[0].shape[1:], inputs[1].shape[1]])


def fit_nn(kind, D, tr, va, sets, y, tag="", verbose=False):
    tf.keras.utils.set_random_seed(SEED)
    prep = Prep(D, tr, kind)
    Xtr, Xva = prep(tr), prep(va)
    model = make_model(kind, Xtr)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy")
    trk = Tracker(y[va], PATIENCE, verbose, tag)

    class CB(tf.keras.callbacks.Callback):
        def on_epoch_end(self, e, logs=None):
            if trk.step(e + 1, self.model.predict(Xva, batch_size=512, verbose=0), self.model.get_weights, f"loss={logs['loss']:.3f}"):
                self.model.stop_training = True

    model.fit(Xtr, y[tr], sample_weight=class_weights(y[tr], 2)[y[tr]], batch_size=BATCH, epochs=EPOCHS, verbose=0, callbacks=[CB()])
    model.set_weights(trk.state)
    outs = [model.predict(prep(s), batch_size=512, verbose=0) if len(s) else np.zeros((0, 2), np.float32) for s in sets]
    return outs, dict(best_epoch=trk.epoch, val_f1=trk.best)


def final_nn(kind, D, rows, y, epochs, d):
    tf.keras.utils.set_random_seed(SEED)
    prep = Prep(D, rows, kind)
    X = prep(rows)
    model = make_model(kind, X)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy")
    model.fit(X, y[rows], sample_weight=class_weights(y[rows], 2)[y[rows]], batch_size=BATCH, epochs=max(epochs, 1), verbose=0)
    model.save(d / "model.keras")
    (d / "model.tflite").write_bytes(tf.lite.TFLiteConverter.from_keras_model(model).convert())
    np.savez(d / "preprocessing.npz", **prep.state())
    (d / "manifest.json").write_text(json.dumps(dict(
        kind=kind, symptom=D["symptom"], feature_set=D["features"], window=WINDOW, epochs=int(epochs), threshold=THRESHOLD,
        sequence_channels=D["channel_names"], feature_names=D["names"],
        note="Features must come from v6_common.window_all exactly as in training."), indent=1))
