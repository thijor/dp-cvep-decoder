import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pyntbci
import pyxdf
from dareplane_utils.signal_processing.filtering import FilterBank
from mnelab.io import read_raw

from cvep_decoder.utils.logging import logger

# from ...dp-speller.speller.speller import TRIAL_TIME, PR, CODE # Imports from another module? probably not the DarePlane way? Another way of ensuring important values such a s trial_time and stimulus presentation are correct?

data_path = Path("./data/training data/sub-P001_ses-S001_task-cvep_run-001_eeg.xdf")
classifier_path = "./data/classifiers"


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


def load_raw_and_events(
    fpath: Path,
    data_stream_name: str = "BioSemi",
    marker_stream_name: str = "KeyboardMarkerStream",
) -> tuple[mne.io.RawArray, np.ndarray]:

    # Load EEG
    data, _ = pyxdf.load_xdf(fpath)
    streams = pyxdf.resolve_streams(fpath)
    names = [stream["name"] for stream in streams]
    raw = read_raw(
        fpath, stream_ids=[streams[names.index(data_stream_name)]["stream_id"]]
    )

    # Align events to the closest time stamp in the data
    evd = data[names.index(marker_stream_name)]
    raw_ts = data[names.index(data_stream_name)]["time_stamps"]
    idx_in_raw = [np.argmin(np.abs(raw_ts - ts)) for ts in evd["time_stamps"]]
    events = np.vstack(
        [idx_in_raw, [e[0] for e in evd["time_series"]]], dtype="object"
    ).T

    return raw, events


def classifier_meta_from_cfg(cfg: dict) -> ClassifierMeta:
    return ClassifierMeta(
        tmin=cfg["tmin"],
        tmax=cfg["tmax"],
        fband=cfg["training"]["passband_hz"],
        n_channels=cfg["n_channels"],
        sfreq=cfg["training"]["target_freq_hz"],
        frame_rate=cfg["frame_rate"],
        selected_channels=cfg["training"]["features"].get("selected_channels", None),
    )


def create_classifier(cfg: dict):
    logger.setLevel(10)
    cmeta = classifier_meta_from_cfg(cfg)

    raw, events = load_raw_and_events(data_path)
    selected_channels = cmeta.selected_channels
    if selected_channels is not None:
        raw.pick_channels(selected_channels)

    # create filterbank
    fb = FilterBank(
        bands={"band": cmeta.fband},
        sfreq=raw.info["sfreq"],
        output="signal",
        n_in_channels=len(raw.info["ch_names"]),
        # n_in_channels=n_channels,
        filter_buffer_s=raw.times[-1],
    )

    data = raw.get_data().T

    fb.filter(data, raw.times)

    data_filtered = fb.get_data()
    raw_f = mne.io.RawArray(data_filtered.T[0], raw.info)

    epo_start_events = {}

    epo = mne.Epochs(
        raw_f,
        events=events,
        tmin=cmeta.tmin - cmeta.pre_trial_window_s,
        tmax=cmeta.tmax,
        baseline=None,
        picks="eeg",
        preload=True,
        verbose=False,
    )
    # need resampling?
    epo.resample(cmeta.sfreq)
    X = epo.get_data(tmin=cmeta.tmin, tmax=cmeta.tmax)

    logger.info(f"X shape: {X.shape} (trials x channels x samples)")

    y = []
    for marker in stream["time_series"]:
        try:
            marker = json.loads(marker[0])
            if isinstance(marker, dict) and "target" in marker:
                y.append(int(marker["target"]))
        except:
            continue

    y = np.array(y)
    logger.info(f"y shape: {y.shape} (trials)")

    # Load codes
    logger.info(f"V shape: {V.shape} (codes x samples)")

    V = np.repeat(
        np.load(cfg["training"]["codes_file"])["codes"].T,
        int(cmeta.sfreq / cmeta.frame_rate),
        axis=1,
    )

    rcca = pyntbci.classifiers.rCCA(
        stimulus=V, fs=fs, event="duration", encoding_length=0.3, onset_event=True
    )

    # Cross-validation
    acc = calc_cv_accuracy(rcca, X, y)
    logger.info(f"Classifier created with accuracy: {acc.mean():.3f}")

    # Full fit for the online use
    rcca.fit(X, y)

    out_file = cfg["training"]["out_file"]
    out_file_meta = cfg["training"]["out_file_meta"]
    with open(out_file, "wb") as file:
        pickle.dump(rcca, file)
    logger.info(f"Classifier saved to {out_file}, meta data saved to {out_file_meta}")

    return 0


def calc_cv_accuracy(
    rcca: pyntbci.classifiers.rCCA, X: np.ndarray, y: np.ndarray, n_folds: int = 4
) -> np.ndarray:
    n_folds = 4
    n_trials = X.shape[0]
    folds = np.repeat(np.arange(n_folds), int(n_trials / n_folds))
    accuracy = np.zeros(n_folds)
    for i_fold in range(n_folds):
        X_trn, y_trn = X[i_fold != folds, :, :], y[i_fold != folds]
        X_tst, y_tst = X[i_fold == folds, :, :], y[i_fold == folds]
        rcca.fit(X_trn, y_trn)
        # TODO: Ask Jordy if rcca.fit would always create a new/fresh classifier? I always make deep copies in CV to ensure fresh setups.
        yh = rcca.predict(X_tst)[:, 0]
        accuracy[i_fold] = np.mean(yh == y_tst)
    return accuracy
