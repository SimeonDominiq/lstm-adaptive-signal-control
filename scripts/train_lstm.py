"""
Stage 7 - train the prediction model and compare it against three baselines.

The model forecasts the eight movement counts for the next 140 s cycle from a
window of recent cycles, as specified in Chapter Three.

Architecture (slide 22): two stacked LSTM layers, dropout, then a dense layer
with eight outputs. Loss MSE, optimiser Adam, early stopping on validation loss.
The input window length N is selected from {5, 10, 15} on the validation set.

Three baselines, each answering a different objection:

  persistence    next cycle = this cycle. If the LSTM cannot beat this, it has
                 learned nothing at all.
  seasonal mean  the average count seen at the same point in the run across the
                 training profiles. If the LSTM cannot beat this, it is only
                 reproducing the shape of a typical day.
  feed-forward   the same windows through a dense network with no recurrence.
                 If the LSTM cannot beat this, the sequence memory is not
                 earning its place and a simpler model would do.

Discipline enforced here, matching the guarantees on slide 17:

  - the scaler is fitted on training inputs only, then applied unchanged
  - sequences never cross a run boundary, so no forecast sees another run
  - chronological order within a run is preserved; a forecast only ever sees
    earlier cycles
  - the window length is chosen on validation, never on test
  - the test partition is evaluated once, at the end

Run after the datasets exist:
    python3 scripts/train_lstm.py
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

MOVEMENTS = ["NB_L", "NB_TR", "SB_L", "SB_TR", "EB_L", "EB_TR", "WB_L", "WB_TR"]
N_MOVEMENTS = len(MOVEMENTS)
WINDOWS = [5, 10, 15]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_partition(name, datadir):
    path = os.path.join(datadir, f"{name}.npz")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}\nRun: python3 scripts/extract_dataset.py")
    z = np.load(path, allow_pickle=True)
    return z["detector"].astype(np.float32)      # (runs, cycles, 8)


def make_sequences(arr, window):
    """
    Build (X, y, cycle_index) with sequences confined to a single run.

    X[i] is the counts for `window` consecutive cycles, y[i] the counts for the
    cycle immediately after. Because each run is processed separately, no window
    ever spans two runs, and every forecast uses only earlier cycles of the same
    run.
    """
    xs, ys, idx = [], [], []
    n_runs, n_cycles, _ = arr.shape
    for r in range(n_runs):
        for t in range(window, n_cycles):
            xs.append(arr[r, t - window:t, :])
            ys.append(arr[r, t, :])
            idx.append(t)
    if not xs:
        raise SystemExit(f"window {window} too long for {n_cycles} cycles")
    return (np.asarray(xs, dtype=np.float32),
            np.asarray(ys, dtype=np.float32),
            np.asarray(idx, dtype=np.int32))


class Scaler:
    """Per-movement standardisation. Fitted on training inputs only."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        flat = X.reshape(-1, X.shape[-1])
        self.mean = flat.mean(axis=0)
        self.std = flat.std(axis=0)
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, A):
        return (A - self.mean) / self.std

    def inverse(self, A):
        return A * self.std + self.mean

    def to_dict(self):
        return dict(mean=self.mean.tolist(), std=self.std.tolist())


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def metrics(y_true, y_pred):
    """
    MAE, RMSE and MAPE.

    MAPE is undefined where the true count is zero, which happens on quiet
    left-turn movements. It is therefore computed only over entries with a
    non-zero true count, and the share of entries used is reported so the
    figure cannot be read as covering everything.
    """
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mask = y_true > 0
    if mask.sum() > 0:
        mape = float(np.mean(np.abs(err[mask] / y_true[mask])) * 100.0)
        cover = float(mask.mean() * 100.0)
    else:
        mape, cover = float("nan"), 0.0
    return dict(MAE=round(mae, 4), RMSE=round(rmse, 4),
                MAPE=round(mape, 2), MAPE_coverage_pct=round(cover, 1))


def per_movement_mae(y_true, y_pred):
    return {m: round(float(np.mean(np.abs(y_pred[:, i] - y_true[:, i]))), 3)
            for i, m in enumerate(MOVEMENTS)}


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------
def persistence(X):
    """Next cycle equals the most recent observed cycle."""
    return X[:, -1, :].copy()


def seasonal_profile(train_arr):
    """Mean count at each cycle index across the training runs."""
    return train_arr.mean(axis=0)          # (cycles, 8)


def seasonal_predict(profile, cycle_idx):
    idx = np.clip(cycle_idx, 0, profile.shape[0] - 1)
    return profile[idx, :]


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def build_lstm(window, units=64, dropout=0.2, seed=7):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window, N_MOVEMENTS)),
        tf.keras.layers.LSTM(units, return_sequences=True),
        tf.keras.layers.LSTM(units),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(N_MOVEMENTS),
    ], name=f"lstm_w{window}")


def build_mlp(window, units=64, dropout=0.2, seed=7):
    """Same inputs, no recurrence - isolates the value of sequence memory."""
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window, N_MOVEMENTS)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units, activation="relu"),
        tf.keras.layers.Dense(units, activation="relu"),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(N_MOVEMENTS),
    ], name=f"mlp_w{window}")


def fit_model(model, Xtr, ytr, Xva, yva, epochs=200, batch=64, patience=15, verbose=0):
    import tensorflow as tf
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                          restore_best_weights=True)
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva),
                     epochs=epochs, batch_size=batch, callbacks=[es],
                     verbose=verbose)
    return model, hist


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--outdir", default=os.path.join(ROOT, "models"))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--verbose", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1,
                    help="train the LSTM and the feed-forward baseline this many "
                         "times with different seeds, and report the spread")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    train_arr = load_partition("train", args.datadir)
    val_arr = load_partition("val", args.datadir)
    test_arr = load_partition("test", args.datadir)
    print(f"loaded  train {train_arr.shape}  val {val_arr.shape}  test {test_arr.shape}")
    print("         (runs, cycles, movements)\n")

    # ---- select the window length on validation only -----------------------
    print("Selecting the input window on the validation set")
    results = {}
    best = None
    for w in WINDOWS:
        Xtr, ytr, _ = make_sequences(train_arr, w)
        Xva, yva, _ = make_sequences(val_arr, w)

        scaler = Scaler().fit(Xtr)                 # training inputs only
        Xtr_s, Xva_s = scaler.transform(Xtr), scaler.transform(Xva)
        ytr_s, yva_s = scaler.transform(ytr), scaler.transform(yva)

        model = build_lstm(w)
        model, hist = fit_model(model, Xtr_s, ytr_s, Xva_s, yva_s,
                                epochs=args.epochs, verbose=args.verbose)
        val_pred = scaler.inverse(model.predict(Xva_s, verbose=0))
        m = metrics(yva, val_pred)
        epochs_run = len(hist.history["loss"])
        results[w] = dict(val=m, epochs=epochs_run)
        print(f"   window {w:>2}  val MAE {m['MAE']:.3f}  RMSE {m['RMSE']:.3f}  "
              f"({epochs_run} epochs)")
        if best is None or m["MAE"] < results[best]["val"]["MAE"]:
            best = w

    print(f"\n   selected window = {best}\n")

    # ---- retrain at the chosen window, then evaluate on test once ----------
    w = best
    Xtr, ytr, _ = make_sequences(train_arr, w)
    Xva, yva, _ = make_sequences(val_arr, w)
    Xte, yte, idx_te = make_sequences(test_arr, w)

    scaler = Scaler().fit(Xtr)
    Xtr_s, Xva_s, Xte_s = (scaler.transform(a) for a in (Xtr, Xva, Xte))
    ytr_s, yva_s = scaler.transform(ytr), scaler.transform(yva)

    seeds = [7 + 100 * k for k in range(max(1, args.repeats))]
    lstm_runs, mlp_runs = [], []
    lstm, hist, lstm_pred, mlp_pred = None, None, None, None

    for k, sd in enumerate(seeds, 1):
        tag = f" (seed {sd}, {k}/{len(seeds)})" if len(seeds) > 1 else ""
        print(f"Training the LSTM at the selected window{tag}")
        m_lstm = build_lstm(w, seed=sd)
        m_lstm, h = fit_model(m_lstm, Xtr_s, ytr_s, Xva_s, yva_s,
                              epochs=args.epochs, verbose=args.verbose)
        p_lstm = scaler.inverse(m_lstm.predict(Xte_s, verbose=0))
        lstm_runs.append(metrics(yte, p_lstm)["MAE"])

        print(f"Training the feed-forward baseline at the same window{tag}")
        m_mlp = build_mlp(w, seed=sd)
        m_mlp, _ = fit_model(m_mlp, Xtr_s, ytr_s, Xva_s, yva_s,
                             epochs=args.epochs, verbose=args.verbose)
        p_mlp = scaler.inverse(m_mlp.predict(Xte_s, verbose=0))
        mlp_runs.append(metrics(yte, p_mlp)["MAE"])

        if k == 1:
            lstm, hist, lstm_pred, mlp_pred = m_lstm, h, p_lstm, p_mlp

    # Per-movement calibration measured on VALIDATION, never on test.
    # The model is fitted on squared error over all movements together, so the
    # small left-turn movements contribute little to the loss and end up
    # systematically under-predicted. Under a proportional-share control law a
    # small absolute under-estimate on a small movement becomes a large
    # relative loss of green, so this bias matters more for control than its
    # size suggests.
    val_pred_cal = scaler.inverse(lstm.predict(Xva_s, verbose=0))
    val_bias = (val_pred_cal - yva).mean(axis=0)
    print("\nPer-movement bias measured on validation (prediction minus actual)")
    for i, m in enumerate(MOVEMENTS):
        print(f"   {m:<6} {val_bias[i]:+.3f}   (mean actual {yva[:, i].mean():.2f})")

    pers_pred = persistence(Xte)
    seas_pred = seasonal_predict(seasonal_profile(train_arr), idx_te)

    # ---- report ------------------------------------------------------------
    table = {
        "LSTM": metrics(yte, lstm_pred),
        "Feed-forward": metrics(yte, mlp_pred),
        "Seasonal mean": metrics(yte, seas_pred),
        "Persistence": metrics(yte, pers_pred),
    }

    print("\nHeld-out test partition (evaluated once)")
    print("-" * 64)
    print(f"{'model':<16}{'MAE':>9}{'RMSE':>9}{'MAPE %':>10}{'MAPE cov %':>12}")
    print("-" * 64)
    for name in ["LSTM", "Feed-forward", "Seasonal mean", "Persistence"]:
        m = table[name]
        print(f"{name:<16}{m['MAE']:>9.3f}{m['RMSE']:>9.3f}"
              f"{m['MAPE']:>10.2f}{m['MAPE_coverage_pct']:>12.1f}")
    print("-" * 64)

    lstm_mae = table["LSTM"]["MAE"]
    print("\nImprovement of the LSTM over each baseline (MAE)")
    verdict = {}
    for name in ["Persistence", "Seasonal mean", "Feed-forward"]:
        b = table[name]["MAE"]
        gain = (b - lstm_mae) / b * 100.0 if b > 0 else float("nan")
        verdict[name] = round(gain, 2)
        mark = "beats" if gain > 0 else "DOES NOT BEAT"
        print(f"   {mark} {name:<14} {gain:+6.2f}%")

    repeat_report = None
    if len(seeds) > 1:
        la, ma = np.array(lstm_runs), np.array(mlp_runs)
        wins = int((la < ma).sum())
        print(f"\nStability across {len(seeds)} training seeds (test MAE)")
        print(f"   LSTM          mean {la.mean():.3f}  sd {la.std(ddof=1):.3f}  "
              f"range {la.min():.3f}-{la.max():.3f}")
        print(f"   Feed-forward  mean {ma.mean():.3f}  sd {ma.std(ddof=1):.3f}  "
              f"range {ma.min():.3f}-{ma.max():.3f}")
        print(f"   LSTM better in {wins} of {len(seeds)} seeds")
        gain = (ma.mean() - la.mean()) / ma.mean() * 100.0
        print(f"   mean improvement over feed-forward: {gain:+.2f}%")
        if wins < len(seeds):
            print("   NOTE: the LSTM does not beat the feed-forward baseline on every")
            print("   seed. The advantage of the recurrence is therefore modest and")
            print("   should be reported with this spread, not as a single figure.")
        repeat_report = dict(
            seeds=seeds,
            lstm_mae=[round(float(x), 4) for x in lstm_runs],
            mlp_mae=[round(float(x), 4) for x in mlp_runs],
            lstm_mean=round(float(la.mean()), 4), lstm_sd=round(float(la.std(ddof=1)), 4),
            mlp_mean=round(float(ma.mean()), 4), mlp_sd=round(float(ma.std(ddof=1)), 4),
            lstm_wins=wins, n_seeds=len(seeds),
            mean_improvement_pct=round(float(gain), 2))

    all_beaten = all(v > 0 for v in verdict.values())
    print("\n" + ("The LSTM improves on all three baselines."
                  if all_beaten else
                  "The LSTM does NOT improve on all three baselines. Chapter Three\n"
                  "commits to accepting the model only if it does. This must be\n"
                  "reported as it stands, not worked around."))

    print("\nPer-movement MAE (LSTM)")
    for m, v in per_movement_mae(yte, lstm_pred).items():
        print(f"   {m:<6} {v}")

    # ---- save --------------------------------------------------------------
    lstm.save(os.path.join(args.outdir, "lstm_model.keras"))
    out = dict(
        window_selected=w,
        window_search={str(k): v for k, v in results.items()},
        scaler=scaler.to_dict(),
        test_metrics=table,
        improvement_over_baselines_pct=verdict,
        accepted=all_beaten,
        per_movement_mae_lstm=per_movement_mae(yte, lstm_pred),
        epochs_trained=len(hist.history["loss"]),
        n_train_sequences=int(Xtr.shape[0]),
        n_val_sequences=int(Xva.shape[0]),
        n_test_sequences=int(Xte.shape[0]),
        seed_stability=repeat_report,
        val_bias=[round(float(b), 5) for b in val_bias],
        val_bias_note=("Per-movement mean of prediction minus actual, measured on "
                       "the validation partition. Available for calibration; "
                       "subtracting it removes the systematic component of the error."),
        note=("Scaler fitted on training inputs only. Sequences never cross a run "
              "boundary. Window chosen on validation. Test evaluated once."),
    )
    jpath = os.path.join(args.outdir, "model_report.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {os.path.relpath(jpath, ROOT)}")
    print(f"wrote {os.path.relpath(os.path.join(args.outdir, 'lstm_model.keras'), ROOT)}")


if __name__ == "__main__":
    main()
