from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re

from dareplane_utils.logging.logger import get_logger
from dareplane_utils.signal_processing.filtering import FilterBank
import joblib
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
    target_accuracy: float = 0.95  # the targeted accuracy used for early stop
    segment_time_s: float = 0.1  # the time used to incrementally grow trials in seconds


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
    events = np.vstack(
        [idx_in_raw, np.asarray([e[0] for e in evd["time_series"]], dtype="object")]
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
    sfreq = int(float(data[names.index(data_stream_name)]["info"]["nominal_srate"][0]))

    return x, events, sfreq


def classifier_meta_from_cfg(cfg: dict) -> ClassifierMeta:
    return ClassifierMeta(
        tmin=cfg["training"]["features"]["tmin_s"],
        tmax=cfg["training"]["features"]["tmax_s"],
        fband=cfg["training"]["features"]["passband_hz"],
        sfreq=cfg["training"]["features"]["target_freq_hz"],
        presentation_rate=cfg["cvep"]["presentation_rate_hz"],
        selected_channels=cfg["training"]["features"].get("selected_channels", None),
        event=cfg["training"]["decoder"]["event"],
        onset_event=cfg["training"]["decoder"]["onset_event"],
        encoding_length=cfg["training"]["decoder"]["encoding_length_s"],
        target_accuracy=cfg["training"]["decoder"]["target_accuracy"],
        segment_time_s=cfg["training"]["decoder"]["segment_time_s"],
    )


def get_training_data_files(cfg: dict) -> list[Path]:
    data_dir = Path(cfg["training"]["data_root"])
    glob_pattern = cfg["training"]["training_files_glob"]

    files = list(data_dir.rglob(glob_pattern))

    if len(files) == 0:
        logger.error(
            f"Did not find files for training at {data_dir} with pattern '{glob_pattern}'"
        )

    return files


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
        cfg["training"]["data_root"] = data_root
    if training_files_glob is not None:
        cfg["training"]["training_files_glob"] = training_files_glob
    if out_file is not None:
        cfg["training"]["out_file"] = out_file
    if out_file_meta is not None:
        cfg["training"]["out_file_meta"] = out_file_meta

    logger.setLevel(10)

    cmeta = classifier_meta_from_cfg(cfg)

    t_files = get_training_data_files(cfg)

    if len(t_files) == 0:
        logger.error("No training files found - stopping fitting attempt")
        return 1

    epo_list = []
    for tfile in t_files:

        x, events, sfreq = load_raw_and_events(
            fpath=tfile,
            data_stream_name=cfg["training"]["features"]["data_stream_name"],
            marker_stream_name=cfg["training"]["features"]["lsl_marker_stream_name"],
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
        epo_events = events[events[:, 1] == cfg["training"]["trial_marker"]]
        n_pre = int(-1 * cmeta.tmin * sfreq)
        n_post = int(cmeta.tmax * sfreq)

        epo_list += [
            e[: (n_post + n_pre), :]  # slice epochs to correct length
            for e in np.split(xf, epo_events[:, 0] - n_pre)[1:]
        ]
    X = np.asarray(epo_list).transpose(0, 2, 1)

    # Resample
    logger.debug(f"The training data X is of shape {X.shape} (n_trials x n_channels x n_samples) before resample")
    X = resample(X, num=int((cmeta.tmax - cmeta.tmin) * cmeta.sfreq), axis=2)
    if np.isnan(X).sum() > 0:
        logger.error("NaNs found after resampling")
    logger.debug(f"The training data X is of shape {X.shape} (n_trials x n_channels x n_samples) after resample")

    # Extract trial labels
    y = np.array(
        [
            int(re.search(r'"target": (\d*)', e[1]).group(1))
            for e in events
            if '"target"' in e[1]
        ]
    )
    logger.debug(f"The labels y is of shape: {y.shape} (n_trials)")

    # Load stimulus sequences
    V = np.repeat(
        np.load(cfg["training"]["codes_file"])["codes"],
        int(cmeta.sfreq / cmeta.presentation_rate),
        axis=1,
    )
    logger.debug(f"The stimulus V is of shape: {V.shape} (n_codes x n_samples)")

    # Fit models of full data
    rcca = fit_rcca_model(cmeta, X, y, V)
    stop = fit_rcca_model_early_stop(cmeta, X, y, V)

    # Cross-validation
    acc, dur = calc_cv_accuracy_early_stop(cmeta, X, y, V)
    logger.info(f"Cross-validation reached accuracy {acc=}, with durations {dur=}")

    out_file = cfg["training"]["out_file"]
    out_file_meta = cfg["training"]["out_file_meta"]

    # assert folders are there
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(out_file_meta).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(rcca, out_file)
    joblib.dump(stop, Path(out_file).with_suffix(".early_stop.joblib"))
    json.dump(asdict(cmeta), open(out_file_meta, "w"))
    logger.info(f"Classifier saved to {out_file}, meta data saved to {out_file_meta}")

    return 0


def fit_rcca_model(
    cmeta: ClassifierMeta,
    X: NDArray,
    y: NDArray,
    V: NDArray,
) -> pyntbci.classifiers.rCCA:
    """
    Fit a standard rCCA model on labeled training data.

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

    Returns
    -------
    rcca: pyntbci.classifiers.rCCA
        A trained rCCA classifier.
    """
    rcca = pyntbci.classifiers.rCCA(
        stimulus=V,
        fs=cmeta.sfreq,
        event=cmeta.event,
        onset_event=cmeta.onset_event,
        encoding_length=cmeta.encoding_length,
    )
    rcca.fit(X, y)
    return rcca


def fit_rcca_model_early_stop(
    cmeta: ClassifierMeta,
    X: NDArray,
    y: NDArray,
    V: NDArray,
) -> pyntbci.stopping.MarginStopping:
    """
    Fit an early stopping rCCA model on labeled training data.

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

    Returns
    -------
    stop: pyntbci.stopping.MarginStopping
        A trained early stopping rCCA classifier.
    """
    rcca = pyntbci.classifiers.rCCA(
        stimulus=V,
        fs=cmeta.sfreq,
        event=cmeta.event,
        onset_event=cmeta.onset_event,
        encoding_length=cmeta.encoding_length,
    )
    stop = pyntbci.stopping.MarginStopping(
        estimator=rcca,
        fs=cmeta.sfreq,
        segment_time=cmeta.segment_time_s,
        target_p=cmeta.target_accuracy,
    )
    stop.fit(X, y)
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

    # Setup classifier
    rcca = pyntbci.classifiers.rCCA(
        stimulus=V,
        fs=cmeta.sfreq,
        event=cmeta.event,
        onset_event=cmeta.onset_event,
        encoding_length=cmeta.encoding_length,
    )

    # Cross-validation
    folds = np.repeat(np.arange(n_folds), int(X.shape[0] / n_folds))
    accuracy = np.zeros(n_folds)

    for i_fold in range(n_folds):

        # Split folds
        X_trn, y_trn = X[i_fold != folds, :, :], y[i_fold != folds]
        X_tst, y_tst = X[i_fold == folds, :, :], y[i_fold == folds]

        # Train classifier and stopping
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

    # Setup classifier
    rcca = pyntbci.classifiers.rCCA(
        stimulus=V,
        fs=cmeta.sfreq,
        event=cmeta.event,
        onset_event=cmeta.onset_event,
        encoding_length=cmeta.encoding_length,
    )
    stop = pyntbci.stopping.MarginStopping(
        estimator=rcca,
        fs=cmeta.sfreq,
        segment_time=cmeta.segment_time_s,
        target_p=cmeta.target_accuracy,
    )

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


if __name__ == "__main__":
    # Add console handler if used as cli
    logger = get_logger("cvep_decoder", add_console_handler=True)

    create_classifier()
