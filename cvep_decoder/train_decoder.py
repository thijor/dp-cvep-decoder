from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re

from dareplane_utils.logging.logger import get_logger
from dareplane_utils.signal_processing.filtering import FilterBank
import joblib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pyntbci
import pyxdf
import toml
from scipy.signal import resample

from cvep_decoder.utils.logging import logger


@dataclass
class ClassifierMeta:
    tmin: float = 0.0  # start of a trial (stimulation) in seconds
    tmax: float = 4.2  # end of a trial (stimulation) in seconds
    fband: tuple[float, float] = (2.0, 30.0)  # passband highpass and lowpass in Hz
    sfreq: float = 120  # EEG sampling frequency in Hz
    presentation_rate: float = 60  # stimulus presentation rate in Hz
    selected_channels: list[str] | None = None  # the EEG channels to use
    pre_trial_window_s: float = 1.0  # baseline window in seconds
    event: str = "contrast"  # the event definition used for rCCA
    onset_event: bool = True  # whether to model an event for the onset of stimulation in each trial in rCCA
    encoding_length: float = 0.3  # the length of the modeled transient response(s) in rCCA
    ctmin: float = 0.0  # lag between LSL marker and hardware marker
    stopping: str = "beta"  # stopping method to use
    segment_time_s: float = 0.1  # the time used to incrementally grow trials in seconds
    target_accuracy: float = 0.95  # the targeted accuracy used for early stop
    min_time: float = 0.1  # the minimum trial time from which decoding can occur
    max_time: float = 4.2  # the maximum trial time at which to force a decoding
    cr: float = 1.0  # cost ratio for Bayesian dynamic stopping
    trained: bool = False  # whether to train distribution stopping


def classifier_meta_from_cfg(cfg: dict) -> ClassifierMeta:
    return ClassifierMeta(
        tmin=cfg["stimulus"]["tmin_s"],
        tmax=cfg["stimulus"]["tmax_s"],
        presentation_rate=cfg["stimulus"]["presentation_rate_hz"],
        fband=cfg["data"]["passband_hz"],
        sfreq=cfg["data"]["target_freq_hz"],
        selected_channels=cfg["data"].get("selected_channels", None),
        event=cfg["decoder"]["event"],
        onset_event=cfg["decoder"]["onset_event"],
        encoding_length=cfg["decoder"]["encoding_length_s"],
        ctmin=cfg["decoder"]["tmin_s"],
        stopping=cfg["decoder"]["stopping"],
        segment_time_s=cfg["decoder"]["segment_time_s"],
        target_accuracy=cfg["decoder"]["target_accuracy"],
        min_time=cfg["decoder"]["min_time_s"],
        max_time=cfg["decoder"]["max_time_s"],
        cr=cfg["decoder"]["cr"],
        trained=cfg["decoder"]["trained"],
    )


def get_training_data_files(cfg: dict) -> list[Path]:
    data_root = Path(cfg["data"]["data_root"])
    glob_pattern = cfg["data"]["training_files_glob"]

    files = list(data_root.rglob(glob_pattern))

    if len(files) == 0:
        logger.error(f"Did not find files for training at {data_root} with pattern '{glob_pattern}'.")
    logger.debug(f"Found {len(files)} files for training with pattern {glob_pattern}.")

    return files


def load_raw_and_events(
    fpath: Path,
    data_stream_name: str = "BioSemi",
    marker_stream_name: str = "cvep-speller-stream",
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
    events = np.vstack([idx_in_raw, np.asarray([e[0] for e in evd["time_series"]], dtype="object")]).T

    # select channels if specified
    if selected_channels is not None:
        if isinstance(selected_channels[0], str):
            ch_names = [
                ch["label"][0]
                for ch in data[names.index(data_stream_name)]["info"]["desc"][0]["channels"][0]["channel"]
            ]
            selected_ch_idx = [ch_names.index(ch) for ch in selected_channels]
        elif isinstance(selected_channels[0], int):
            selected_ch_idx = selected_channels
        else:
            raise ValueError(f"{selected_channels=} must be a list of `str` or `int` or `None`.")
        x = x[:, selected_ch_idx]

    sfreq = int(float(data[names.index(data_stream_name)]["info"]["nominal_srate"][0]))

    return x, events, sfreq


# all options beyond cfg are for overwriting the config if necessary
def create_classifier(
    data_root: Path | None = None,
    training_files_glob: str | None = None,
    out_file: Path | None = None,
    out_file_meta: Path | None = None,
) -> int:

    cfg = toml.load("./configs/decoder.toml")
    # Apply overwrites
    if data_root is not None:
        cfg["data"]["data_root"] = data_root
    if training_files_glob is not None:
        cfg["data"]["training_files_glob"] = training_files_glob
    if out_file is not None:
        cfg["decoder"]["decoder_file"] = out_file
    if out_file_meta is not None:
        cfg["decoder"]["decoder_meta_file"] = out_file_meta

    logger.setLevel(10)

    cmeta = classifier_meta_from_cfg(cfg)

    t_files = get_training_data_files(cfg)

    if len(t_files) == 0:
        logger.error("No training files found - stopping fitting attempt")
        return 1

    eeg_list = []
    lbl_list = []
    for t_file in t_files:

        # Load raw continuous data
        x, events, sfreq = load_raw_and_events(
            fpath=t_file,
            marker_stream_name=cfg["streams"]["marker_stream_name"],
            data_stream_name=cfg["streams"]["data_stream_name"],
            selected_channels=cfg["data"].get("selected_channels", None),
        )

        # Bandpass filter
        fb = FilterBank(
            bands={"band": cmeta.fband},
            sfreq=sfreq,
            output="signal",
            n_in_channels=x.shape[1],
            filter_buffer_s=np.ceil(x.shape[0] / sfreq),
        )
        fb.filter(x, np.arange(x.shape[0]) / sfreq)
        xf = fb.get_data()[:, :, 0]

        # Extract marker onsets
        onsets = events[events[:, 1] == cfg["stimulus"]["trial_marker"], 0]

        # Add padding interval to catch filtering artefacts
        if cfg["streams"]["padding_size_s"] is not None and cfg["streams"]["padding_size_s"] > 0:
            pad = int(cfg["streams"]["padding_size_s"] * sfreq)
            onsets -= pad

        # Slice data to trials including padding interval to remove filtering artefacts
        eeg_list += [
            xf[t - int(cmeta.tmin * sfreq):t + int(cmeta.tmax * sfreq), :]
            for t in onsets
        ]

        # Extract trial labels
        lbl_list += [
            int(m.split(";")[1].split("=")[1])
            for m in events[:, 1]
            if m.startswith(cfg["stimulus"]["cue_marker"])]

    # Concatenate trials
    X = np.stack(eeg_list, axis=0).transpose(0, 2, 1)  # trials, channels, samples
    y = np.stack(lbl_list, axis=0)  # trials

    # Resample
    X = resample(X, num=int((cmeta.tmax - cmeta.tmin) * cmeta.sfreq), axis=2)
    if np.isnan(X).sum() > 0:
        logger.error("NaNs found after resampling")

    # Remove padding interval to catch filtering artefacts
    if cfg["streams"]["padding_size_s"] is not None and cfg["streams"]["padding_size_s"] > 0:
        pad = int(cfg["streams"]["padding_size_s"] * cmeta.sfreq)
        X = X[:, :, pad:]

    # Load stimulus sequences
    V = np.repeat(
        np.load(cfg["training"]["codes_file"])["codes"],
        int(cmeta.sfreq / cmeta.presentation_rate),
        axis=1,
    )

    logger.debug(f"The data X are of shape {X.shape} (trials x channels x samples)")
    logger.debug(f"The labels y are of shape {y.shape} (trials)")
    logger.debug(f"The stimuli V are of shape: {V.shape} (codes x samples)")

    # Fit model on training data
    model = get_rcca_model_early_stop(cmeta, V)
    model.fit(X, y)

    # Cross-validation for performance estimation
    n_folds = 4
    acc, dur = calc_cv_accuracy_early_stop(cmeta, X, y, V, n_folds)
    logger.info(f"Cross-validated accuracy of {np.mean(acc):.3f} +/- {np.std(acc):.3f}")
    logger.info(f"Cross-validated duration of {np.mean(dur):.2f} +/- {np.std(dur):.2f}")

    # Swap out codes file if we have a different file selected for the online phase.
    if cfg["online"]["codes_file"] != cfg["training"]["codes_file"]:
        V = np.repeat(
            np.load(cfg["online"]["codes_file"])["codes"],
            int(cmeta.sfreq / cmeta.presentation_rate),
            axis=1,
        )
        logger.info("Different codeset for training and online phase detected.")
        logger.debug(f"New stimuli V are of shape: {V.shape} (codes x samples)")
        model.estimator.set_stimulus(V)

    # Optimize subset of codes
    n_keys = cfg["stimulus"]["n_keys"]
    if n_keys != 0 and n_keys < V.shape[0]:
        Ts = model.estimator.get_T(V.shape[1])[:, 0, :]  # select component
        subset = pyntbci.stimulus.optimize_subset_clustering(Ts, n_keys)
        logger.debug(f"Created optimal subset for {n_keys} keys using {len(Ts)} codes")
    else:
        subset = np.array([i for i in range(n_keys)])  # Mockup "subset" which is just the 0:n_keys-1.
        logger.debug("Skipped optimal subset (Number of keys equals number of codes or Number of keys is set to 0)")
    V = V[subset, :]  # select optimal code subset
    model.estimator.set_stimulus(V)

    # Hardcoded dictionaries containing a key:[n_neighbours] relationship, with only east and south neighbours
    # TODO Should probably just store this in a JSON or ideally come-up with some non-hardcoded method.
    keyboard_dict = {
        0:  [1, 13],
        1:  [2, 13, 14],
        2:  [3, 14, 15],
        3:  [4, 15, 16],
        4:  [5, 16, 17],
        5:  [6, 17, 18],
        6:  [7, 18, 19],
        7:  [8, 19, 20],
        8:  [9, 20, 21],
        9:  [10, 21, 22],
        10: [11, 22, 23],
        11: [22, 23, 24],
        12: [24],
        13: [14, 25, 26],
        14: [15, 26, 27],
        15: [16, 27, 28],
        16: [17, 28, 29],
        17: [18, 29, 30],
        18: [19, 30, 31],
        19: [20, 31, 32],
        20: [21, 32, 33],
        21: [22, 33, 34],
        22: [23, 34, 35],
        23: [24, 35, 36],
        24: [36, 37],
        25: [26, 38],
        26: [27, 38, 39],
        27: [28, 39, 40],
        28: [29, 40, 41],
        29: [30, 41, 42],
        30: [31, 42, 43],
        31: [32, 43, 44],
        32: [33, 44, 45],
        33: [34, 45, 46],
        34: [35, 46, 47],
        35: [36, 47, 48],
        36: [37, 48],
        37: [],
        38: [39],
        39: [40],
        40: [41],
        41: [42, 49],
        42: [43, 49, 50],
        43: [44, 50, 51],
        44: [45, 51, 52],
        45: [46, 52],
        46: [47],
        47: [48],
        48: [],
        49: [50],
        50: [51],
        51: [52],
        52: [],
    }
    if n_keys == 53:
        pass
    else:
        keyboard_dict = dict()

    # Convert the hard-coded dict into nd.array of shape (neighbours, 2)
    neighbour_set = []
    for key, neighbours in keyboard_dict.items():
        for neighbour in neighbours:
            neighbour_set.append([key, neighbour])
    neighbours = np.array(neighbour_set)

    # Optimal layout of codes
    Ts = model.estimator.get_T(V.shape[1])[:, 0, :]  # select component
    layout = pyntbci.stimulus.optimize_layout_incremental(Ts, neighbours)
    V = V[layout, :]  # order codes with optimal layout
    model.estimator.set_stimulus(V)
    logger.debug(f"Created optimal layout for {n_keys} keys using {len(Ts)} codes")

    # Save classifier
    out_file = cfg["decoder"]["decoder_file"]
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_file)
    logger.info(f"Classifier saved to {out_file}")

    # Save classifier meta
    out_file_meta = cfg["decoder"]["decoder_meta_file"]
    Path(out_file_meta).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file_meta, "w") as fid:
        json.dump(asdict(cmeta), fid)
        logger.info(f"Classifier meta data saved to {out_file_meta}")

    # Save the optimal subset and layout
    json_data = {
        "codes_file": Path(cfg["online"]["codes_file"]).name,
        "subset": subset.tolist(),
        "layout": layout.tolist(),
    }
    out_file = cfg["decoder"]["decoder_subset_layout_file"]
    with open(out_file, "w") as fid:
        json.dump(json_data, fid)
        logger.info(f"Subset and layout saved to {out_file}")

    # Visualize classifier
    plot_rcca_model_early_stop(model, acc, dur, n_folds, cfg)
    plt.show()  # halts here until the figure is closed

    return 0


def get_rcca_model(
    cmeta: ClassifierMeta,
    V: NDArray,
) -> pyntbci.classifiers.rCCA:
    """
    Fit a standard rCCA model on labeled training data.

    Parameters
    ----------
    cmeta: ClassifierMeta
        The classifier hyperparameters.
    V: NDArray
        The stimulus matrix of shape (n_codes x n_samples).

    Returns
    -------
    rcca: pyntbci.classifiers.rCCA
        An untrained trained rCCA classifier.
    """
    rcca = pyntbci.classifiers.rCCA(
        stimulus=V,
        fs=int(cmeta.sfreq),
        event=cmeta.event,
        onset_event=cmeta.onset_event,
        encoding_length=cmeta.encoding_length,
        tmin=cmeta.ctmin,
    )
    return rcca


def get_rcca_model_early_stop(
    cmeta: ClassifierMeta,
    V: NDArray,
) -> pyntbci.stopping.MarginStopping | pyntbci.stopping.DistributionStopping | pyntbci.stopping.CriterionStopping:
    """
    Fit an early stopping rCCA model on labeled training data.

    Parameters
    ----------
    cmeta: ClassifierMeta
        The classifier hyperparameters.
    V: NDArray
        The stimulus matrix of shape (n_codes x n_samples).

    Returns
    -------
    stop: pyntbci.stopping.MarginStopping | pyntbci.stopping.DistributionStopping | pyntbci.stopping.CriterionStopping
        An untrained early stopping rCCA classifier.
    """
    rcca = get_rcca_model(cmeta, V)
    if cmeta.stopping == "margin":
        stop = pyntbci.stopping.MarginStopping(
            estimator=rcca,
            fs=int(cmeta.sfreq),
            segment_time=cmeta.segment_time_s,
            target_p=cmeta.target_accuracy,
            min_time=cmeta.min_time,
            max_time=cmeta.max_time,
        )
    elif cmeta.stopping in ["beta", "norm"]:
        stop = pyntbci.stopping.DistributionStopping(
            estimator=rcca,
            fs=int(cmeta.sfreq),
            segment_time=cmeta.segment_time_s,
            distribution=cmeta.stopping,
            target_p=cmeta.target_accuracy,
            min_time=cmeta.min_time,
            max_time=cmeta.max_time,
            trained=cmeta.trained,
        )
    elif cmeta.stopping == "accuracy":
        stop = pyntbci.stopping.CriterionStopping(
            estimator=rcca,
            fs=int(cmeta.sfreq),
            segment_time=cmeta.segment_time_s,
            criterion=cmeta.stopping,
            target=cmeta.target_accuracy,
            min_time=cmeta.min_time,
            max_time=cmeta.max_time,
        )
    elif cmeta.stopping in ["bds0", "bds1", "bds2"]:
        stop = pyntbci.stopping.BayesStopping(
            estimator=rcca,
            fs=int(cmeta.sfreq),
            segment_time=cmeta.segment_time_s,
            method=cmeta.stopping,
            cr=cmeta.cr,
            min_time=cmeta.min_time,
            max_time=cmeta.max_time,
        )
    else:
        ValueError(f"Unknown stopping method: {cmeta.stopping}")
    return stop


def calc_cv_accuracy(
    cmeta: ClassifierMeta,
    X: NDArray,
    y: NDArray,
    V: NDArray,
    n_folds: int = 4,
) -> NDArray:
    """
    Evaluate an rCCA model on labeled training data using k-fold cross-validation.

    Parameters
    ----------
    cmeta: ClassifierMeta
        The classifier hyperparameters.
    X: NDArray
        The EEG data matrix of shape (n_trials x n_channels x n_samples).
    y: NDArray
        The label vector of shape (n_trials).
    V: NDArray
        The stimulus matrix of shape (n_codes x n_samples).
    n_folds: int (default: 4)
        The number of folds for cross-validation.

    Returns
    -------
    accuracy: NDArray
        The vector of accuracies for each of the folds of shape (n_folds).
    """
    folds = np.repeat(np.arange(n_folds), int(X.shape[0] / n_folds))
    accuracy = np.zeros(n_folds)
    for i_fold in range(n_folds):

        # Split folds
        X_trn, y_trn = X[i_fold != folds, :, :], y[i_fold != folds]
        X_tst, y_tst = X[i_fold == folds, :, :], y[i_fold == folds]

        # Train classifier and stopping
        rcca = get_rcca_model(cmeta, V)
        rcca.fit(X_trn, y_trn)
        yh = rcca.predict(X_tst)
        accuracy[i_fold] = np.mean(yh == y_tst)

    return accuracy


def calc_cv_accuracy_early_stop(
    cmeta: ClassifierMeta,
    X: NDArray,
    y: NDArray,
    V: NDArray,
    n_folds: int = 4,
) -> tuple[NDArray, NDArray]:
    """
    Evaluate an early stopping rCCA model on labeled training data using k-fold cross-validation.

    Parameters
    ----------
    cmeta: ClassifierMeta
        The classifier hyperparameters.
    X: NDArray
        The EEG data matrix of shape (n_trials x n_channels x n_samples).
    y: NDArray
        The label vector of shape (n_trials).
    V: NDArray
        The stimulus matrix of shape (n_codes x n_samples).
    n_folds: int (default: 4)
        The number of folds for cross-validation.

    Returns
    -------
    accuracy: NDArray
        The vector of accuracies for each of the folds of shape (n_folds).
    duration: NDArray
        The vector of trial durations for each of the folds of shape (n_folds).
    """

    # Cross-validation
    folds = np.repeat(np.arange(n_folds), int(np.ceil(X.shape[0] / n_folds)))[
        : X.shape[0]
    ]
    accuracy = np.zeros(n_folds)
    duration = np.zeros(n_folds)

    for i_fold in range(n_folds):

        # Split folds
        X_trn, y_trn = X[i_fold != folds, :, :], y[i_fold != folds]
        X_tst, y_tst = X[i_fold == folds, :, :], y[i_fold == folds]

        # Train classifier and stopping
        stop = get_rcca_model_early_stop(cmeta, V)
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

    return accuracy, duration


def plot_rcca_model_early_stop(stop, acc, dur, n_folds, cfg):
    fig, ax = plt.subplots(2, 2, figsize=(11.69, 5))

    # Transient response(s)
    r = stop.estimator.r_.reshape((len(stop.estimator.events_), -1)).T
    ax[0, 0].plot(np.arange(r.shape[0]) / stop.fs, r)
    ax[0, 0].set_xlabel("time [s]")
    ax[0, 0].set_ylabel("amplitude [a.u.]")
    ax[0, 0].legend(stop.estimator.events_, bbox_to_anchor=(1.0, 1.0))
    ax[0, 0].grid("both", alpha=0.1, color="k")
    ax[0, 0].set_title("temporal response(s)")

    # Spatial filter
    if cfg["data"]["capfile"] == "":
        ax[0, 1].plot(1 + np.arange(stop.estimator.w_.size), stop.estimator.w_)
        ax[0, 1].set_xlabel("electrode")
        ax[0, 1].set_ylabel("weight [a.u.]")
    else:
        pyntbci.plotting.topoplot(stop.estimator.w_, locfile=cfg["data"]["capfile"], ax=ax[0, 1])
    ax[0, 1].set_title("spatial filter")

    # Stopping
    if isinstance(stop, pyntbci.stopping.MarginStopping):
        ax[1, 0].plot(np.arange(stop.margins_.size) * stop.segment_time, stop.margins_, label="threshold")
        ax[1, 0].set_ylim([-0.05, 1.05])
        ax[1, 0].set_xlabel("time [s]")
        ax[1, 0].set_ylabel("margin")
        ax[1, 0].legend(bbox_to_anchor=(1.0, 1.0))
        ax[1, 0].grid("both", alpha=0.1, color="k")
        ax[1, 0].set_title("stopping margins")
    else:
        ax[1, 0].set_axis_off()

    # Cross-validated accuracy (and duration)
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    ax[1, 1].set_xlim([0, 1])
    ax[1, 1].set_ylim([0, 1])
    ax[1, 1].text(0.1, 0.6, f"Accuracy: {acc.mean():.3f}")
    ax[1, 1].text(0.1, 0.4, f"Duration: {dur.mean():.3f}")
    ax[1, 1].set_title(f"{n_folds:d}-fold cross-validation")

    fig.tight_layout()
    fig.canvas.manager.set_window_title("Calibrated classifier: close figure to continue")


if __name__ == "__main__":
    # Add console handler if used as cli
    logger = get_logger("cvep_decoder", add_console_handler=True)

    create_classifier()
