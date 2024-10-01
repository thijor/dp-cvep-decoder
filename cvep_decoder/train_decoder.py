import json
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pyntbci
import pyxdf
import toml
from dareplane_utils.logging.logger import get_logger
from dareplane_utils.signal_processing.filtering import FilterBank
from scipy.signal import resample

from cvep_decoder.utils.logging import logger


@dataclass
class ClassifierMeta:
    tmin: float = 0.0
    tmax: float = 4.2
    fband: tuple[float, float] = (2.0, 30.0)
    n_channels = 32
    sfreq: float = 120
    frame_rate: float = 60
    pre_trial_window_s: float = 1.0
    code: str = "mgold_61_6521"
    selected_channels: list[str] | None = None
    segment_time_s: float = 0.1
    event: str = "contrast"
    onset_event: bool = True
    encoding_length: float = 0.3
    target_accuracy: float = 0.95  # used for early stop


def load_raw_and_events(
    fpath: Path,
    data_stream_name: str = "BioSemi",
    marker_stream_name: str = "KeyboardMarkerStream",
    selected_channels: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:

    # Load EEG
    data, _ = pyxdf.load_xdf(fpath)
    streams = pyxdf.resolve_streams(fpath)
    names = [stream["name"] for stream in streams]
    x = data[names.index(data_stream_name)]["time_series"]

    # Align events to the closest time stamp in the data
    evd = data[names.index(marker_stream_name)]
    raw_ts = data[names.index(data_stream_name)]["time_stamps"]
    idx_in_raw = [np.argmin(np.abs(raw_ts - ts)) for ts in evd["time_stamps"]]
    events = np.vstack(
        [idx_in_raw, [e[0] for e in evd["time_series"]]], dtype="object"
    ).T

    # select channels if specified
    if selected_channels is not None:
        ch_names = [
            ch["label"][0]
            for ch in data[names.index(data_stream_name)]["info"]["desc"][0][
                "channels"
            ][0]["channel"]
        ]
        idx = [ch_names.index(ch) for ch in selected_channels]
        x = x[:, idx]
    sfreq = int(data[names.index(data_stream_name)]["info"]["nominal_srate"][0])

    return x, events, sfreq


def classifier_meta_from_cfg(cfg: dict) -> ClassifierMeta:
    return ClassifierMeta(
        tmin=cfg["training"]["features"]["tmin_s"],
        tmax=cfg["training"]["features"]["tmax_s"],
        fband=cfg["training"]["features"]["passband_hz"],
        sfreq=cfg["training"]["features"]["target_freq_hz"],
        frame_rate=cfg["cvep"]["frame_rate_hz"],
        selected_channels=cfg["training"]["features"].get("selected_channels", None),
    )


def get_training_data_files(cfg: dict) -> list[Path]:
    files = list(
        Path(cfg["training"]["data_root"]).rglob(cfg["training"]["training_files_glob"])
    )

    return files


def create_classifier(cfg: dict):
    logger.setLevel(10)

    cfg = toml.load("./configs/decoder.toml")
    cmeta = classifier_meta_from_cfg(cfg)

    t_files = get_training_data_files(cfg)
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
            sfreq=cmeta.sfreq,
            output="signal",
            n_in_channels=x.shape[1],
            filter_buffer_s=np.ceil(x.shape[0] / sfreq),
        )

        fb.filter(x, np.arange(x.shape[0]) / sfreq)

        xf = fb.get_data()[:, :, 0]

        # Slice data to trials
        epo_events = events[events[:, 1] == "start_trial"]
        n_pre = int(-1 * cmeta.tmin * sfreq)
        n_post = int(cmeta.tmax * sfreq)

        epo_list += [
            e[: (n_post + n_pre), :]  # slice epochs to correct lenght
            for e in np.split(xf, epo_events[:, 0] - n_pre)[1:]
        ]

    X = np.asarray(epo_list).transpose(0, 2, 1)

    logger.debug(f"The training data is of shape {X.shape} before resample")

    # Resample
    X = resample(X, num=int((cmeta.tmax - cmeta.tmin) * cmeta.sfreq), axis=2)
    logger.debug(f"The training data is of shape {X.shape} after resample")

    V = np.repeat(
        np.load(cfg["training"]["codes_file"])["codes"].T,
        int(cmeta.sfreq / cmeta.frame_rate),
        axis=1,
    )

    logger.info(f"V shape: {V.shape} (codes x samples)")

    y = np.array(
        [
            int(re.search(r'"target": (\d*)', e[1]).group(1))
            for e in events
            if '"target"' in e[1]
        ]
    )

    rcca = fit_rcca_model(cmeta, X, y, V)

    # Cross-validation
    acc, dur = calc_cv_accuracy(rcca, cmeta, V, X, y)

    out_file = cfg["training"]["out_file"]
    out_file_meta = cfg["training"]["out_file_meta"]

    joblib.save(rcca, out_file)
    # json.dump(ClassifierMeta, out_file_meta)  # TODO: Fix storing the meta
    logger.info(f"Classifier saved to {out_file}, meta data saved to {out_file_meta}")

    return 0


def fit_rcca_model(
    cmeta: dict, X: np.ndarray, y: np.ndarray, V: np.ndarray
) -> pyntbci.classifiers.rCCA:
    rcca = pyntbci.classifiers.rCCA(
        stimulus=V,
        fs=cmeta.sfreq,
        event="duration",
        encoding_length=cmeta.encoding_length,
        onset_event=True,
    )
    rcca.fit(X, y)
    return rcca


def calc_cv_accuracy(
    rcca: pyntbci.classifiers.rCCA,
    cmeta: ClassifierMeta,
    V: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 4,
) -> tuple[np.ndarray, np.ndarray]:

    # Setup classifier
    rcca = pyntbci.classifiers.rCCA(
        stimulus=V,
        fs=cmeta.sfreq,
        event=cmeta.event,
        onset_event=True,
        encoding_length=cmeta.encoding_length,
    )
    stop = pyntbci.stopping.MarginStopping(
        estimator=rcca,
        segment_time=cmeta.segment_time_s,
        fs=cmeta.sfreq,
        target_p=cmeta.target_accuracy,
    )

    # Cross-validation
    folds = np.repeat(np.arange(n_folds), int(X.shape[0] / n_folds))
    accuracy = np.zeros(n_folds)
    duration = np.zeros(n_folds)

    for i_fold in range(n_folds):

        # Split folds
        X_trn, y_trn = X[i_fold != folds, :, :], y[i_fold != folds]
        X_tst, y_tst = X[i_fold == folds, :, :], y[i_fold == folds]

        # Train classifier and stopping
        stop.fit(X_trn, y_trn)

        # Loop trials
        yh_tst = np.zeros(y_tst.size)
        yh_dur = np.zeros(y_tst.size)
        for i_trial in range(y_tst.size):
            X_i = X_tst[[i_trial], :, :]

            # Loop segments
            for i_segment in range(
                int(X.shape[2] / (cmeta.segment_time_s * cmeta.sfreq))
            ):

                # Apply classifier
                label = stop.predict(
                    X_i[
                        :,
                        :,
                        : int((1 + i_segment) * cmeta.segment_time_s * cmeta.sfreq),
                    ]
                )[0]

                # Stop the trial if classified
                if label >= 0:
                    yh_tst[i_trial] = label
                    yh_dur[i_trial] = (1 + i_segment) * cmeta.segment_time_s
                    break

        # Compute performance
        accuracy[i_fold] = np.mean(yh_tst == y_tst)
        duration[i_fold] = np.mean(yh_dur)

    logger.info(
        f"Cross-validated accuracy of {np.mean(accuracy):.3f} +/- {np.std(accuracy):.3f}"
    )
    logger.info(
        f"Cross-validated duration of {np.mean(duration):.2f} +/- {np.std(duration):.2f}"
    )

    return accuracy, duration


def calc_cv_accuracy_no_early_stop(
    rcca: pyntbci.classifiers.rCCA,
    cmeta: ClassifierMeta,
    V: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 4,
) -> np.ndarray:

    # Setup classifier
    rcca = pyntbci.classifiers.rCCA(
        stimulus=V,
        fs=cmeta.sfreq,
        event=cmeta.event,
        onset_event=True,
        encoding_length=cmeta.encoding_length,
    )

    # Cross-validation
    folds = np.repeat(np.arange(n_folds), int(X.shape[0] / n_folds))
    accuracy = np.zeros(n_folds)
    duration = np.zeros(n_folds)

    for i_fold in range(n_folds):

        # Split folds
        X_trn, y_trn = X[i_fold != folds, :, :], y[i_fold != folds]
        X_tst, y_tst = X[i_fold == folds, :, :], y[i_fold == folds]

        # Train classifier and stopping
        rcca.fit(X_trn, y_trn)
        yh = rcca.predict(X_tst)
        accuracy[i_fold] = np.mean(yh == y_tst)

    return accuracy
