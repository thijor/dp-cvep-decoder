# With the provided training data, we expect a classifier accuracy for 100%
# Note: these tests are rather regression test matching the old code
# than actual unit tests

import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pyntbci
import toml
from dareplane_utils.signal_processing.filtering import FilterBank
from scipy.signal import resample

from cvep_decoder.train_decoder import (calc_cv_accuracy_no_early_stop,
                                        classifier_meta_from_cfg,
                                        fit_rcca_model, load_raw_and_events)


def test_epoch_slicing():
    cfg = toml.load("./configs/decoder.toml")
    cmeta = classifier_meta_from_cfg(cfg)
    cmeta.fband = [2, 30]

    t_files = ["./tests/assets/raw_eeg.xdf"]

    epo_list = []
    for tfile in t_files:

        x, events, sfreq = load_raw_and_events(
            fpath=tfile,
            data_stream_name=cfg["training"]["features"]["data_stream_name"],
            marker_stream_name=cfg["training"]["features"]["marker_stream_name"],
            selected_channels=cfg["training"]["features"].get(
                "selected_channels", None
            ),
        )

        # create filterbank
        fb = FilterBank(
            bands={"band": cmeta.fband},
            sfreq=sfreq,
            output="signal",
            n_in_channels=x.shape[1],
            filter_buffer_s=np.ceil(x.shape[0] / sfreq),
        )

        fb.filter(x, np.arange(x.shape[0]) / sfreq)

        xf = fb.get_data()[:, :, 0]

        # Slice data to trials
        epo_events = events[events[:, 1] == "start_trial"]

        # Note that the test data was generated from hardware markers
        # The first epoch markers should be at the following raw indeces
        #   [  3985,      0,      1],
        #   [  7364,      0,      1],
        #   [ 10745,      0,      1]
        #
        #   This is an offset of 21 samples
        epo_events[:, 0] += 21

        n_pre = int(-1 * cmeta.tmin * sfreq)
        n_post = int(cmeta.tmax * sfreq)

        epo_list += [
            e[: (n_post + n_pre), :]  # slice epochs to correct lenght
            for e in np.split(xf, epo_events[:, 0] - n_pre)[1:]
        ]

    X = np.asarray(epo_list).transpose(0, 2, 1)
    X = resample(X, num=int((cmeta.tmax - cmeta.tmin) * cmeta.sfreq), axis=2)

    Xt = (
        np.load("./tests/assets/X.npy") * 1e6
    )  # created with help of mne -> scale to uV

    # Note: There is a small difference in the epoched data expected
    # Use this plot to visualize
    # plot_epochs_compare(X, Xt)
    assert (X - Xt).mean() < 0.01

    # also test labels
    y = np.array(
        [
            int(re.search(r'"target": (\d*)', e[1]).group(1))
            for e in events
            if '"target"' in e[1]
        ]
    )

    yt = np.load("./tests/assets/y.npy")
    assert np.all(y == yt)


def test_loading_code():
    cfg = toml.load("./configs/decoder.toml")
    cmeta = classifier_meta_from_cfg(cfg)
    # the parameters used from the Vt
    cmeta.sfreq = 120
    cmeta.frame_rate = 60

    V = np.repeat(
        np.load("./tests/assets/mgold_61_6521.npz")["codes"].T,
        int(cmeta.sfreq / cmeta.frame_rate),
        axis=1,
    )

    Vt = np.load("./tests/assets/V.npy")
    assert np.all(V == Vt)


def test_classifier_and_cv_acc():
    X = np.load("./tests/assets/X.npy")
    y = np.load("./tests/assets/y.npy")
    V = np.load("./tests/assets/V.npy")

    cfg = toml.load("./configs/decoder.toml")
    cmeta = classifier_meta_from_cfg(cfg)
    cmeta.sfreq = 120
    cmeta.encoding_length = 0.3

    rcca = fit_rcca_model(cmeta, X, y, V)

    rcca_t = joblib.load("./tests/assets/rcca_test_mgold_61.joblib")

    assert np.all(rcca.w_ == rcca_t.w_)
    assert np.all(rcca.r_ == rcca_t.r_)
    assert np.all(rcca.Ts_ == rcca_t.Ts_)
    assert np.all(rcca.Tw_ == rcca_t.Tw_)

    # Cross-validation - the cross val function will consider early stopping
    # -> tune up the target_accuracy to avoid
    cmeta.target_accuracy = 1.0
    cv_acc = calc_cv_accuracy_no_early_stop(rcca, cmeta, V, X, y)

    n_folds = 4
    n_trials = X.shape[0]
    folds = np.repeat(np.arange(n_folds), int(n_trials / n_folds))

    accuracy = np.zeros(n_folds)
    rcca = pyntbci.classifiers.rCCA(
        stimulus=V, fs=120, event="duration", encoding_length=0.3, onset_event=True
    )
    for i_fold in range(n_folds):
        X_trn, y_trn = X[i_fold != folds, :, :], y[i_fold != folds]
        X_tst, y_tst = X[i_fold == folds, :, :], y[i_fold == folds]

        rcca.fit(X_trn, y_trn)

        yh = rcca.predict(X_tst)
        accuracy[i_fold] = np.mean(yh == y_tst)

    assert np.all(cv_acc == accuracy)


def plot_epochs_compare(X, Xt):
    import plotly.express as px
    import polars as pl

    df = (
        pl.DataFrame(
            {
                "X_0_ch1": X[0, 0, :],
                "Xt_0_ch1": Xt[0, 0, :],
                "X_1_ch1": X[1, 0, :],
                "Xt_1_ch1": Xt[1, 0, :],
                "idx": np.arange(X.shape[2]),
            }
        )
        .melt(id_vars=["idx"])
        .with_columns(
            src=pl.col("variable").str.extract(r"([^_]*).*"),
            ch=pl.col("variable").str.extract(r"_([^_]*)_"),
        )
    )

    fig = px.line(
        df,
        x="idx",
        y="value",
        facet_row="ch",
        color="src",
        template="plotly_white",
    )
    fig.show()
