import json
import os
from pathlib import Path

from dareplane_utils.signal_processing.filtering import FilterBank
from dareplane_utils.logging.logger import get_logger
import joblib
import numpy as np
import pyntbci
import pyxdf
from scipy.stats import resample

from cvep_decoder.utils.logging import logger


DATA_DIR = "./data"
DATA_FILE = "sub-001_ses-001_task-cvep_run-001_eeg.xdf"

CLASSIFIER_DIR = "./data"
CLASSIFIER_FILE = "sub-001_ses-001_task-cvep_run-001_classifier.joblib"

CODE_DIR = "."
CODE_FILE = "mgold_61_6521.npz"

EEG_LSL_STREAM_NAME = "BioSemi"
MARKER_LSL_STREAM_NAME = "marker-stream"

SPELLER_MARKER_TRIAL_START = "start_trial"

SEGMENT_TIME = 0.1
TRIAL_TIME = 4.2

PR = 60
FS = 120
L_FREQ = 2.0
H_FREQ = 30.0
N_CHANNELS = 32

N_FOLDS = 4

EVENT = "contrast"
ONSET_EVENT = True
ENCODING_LENGTH = 0.3
TARGETED_ACCURACY = 0.95


def create_classifier(log_level: int = 30):
    logger = get_logger("cvep_decoder", add_console_handler=True)
    logger.setLevel(log_level)

    # Load training file
    fn = os.path.join(Path(DATA_DIR) / DATA_FILE)
    streams = pyxdf.load_xdf(fn)[0]
    stream_names = [stream["info"]["name"][0] for stream in streams]
    logger.debug(f"Data loaded from {Path(DATA_DIR) / DATA_FILE}")

    eeg_stream = streams[stream_names.index(EEG_LSL_STREAM_NAME)]
    sfreq = float(eeg_stream["info"]["nominal_srate"][0])
    data = eeg_stream["time_series"][:, 1:N_CHANNELS]
    time = eeg_stream["time_series"] - eeg_stream["time_series"][0]

    # Bandpass filter
    fb = FilterBank(
        bands={"band": (L_FREQ, H_FREQ)},
        sfreq=sfreq,
        output="signal",
        n_in_channels=N_CHANNELS,
        filter_buffer_s=time[-1],
    )
    fb.filter(data, time)
    data = fb.get_data()

    # Extract trial onsets and labels
    marker_stream = streams[stream_names.index(MARKER_LSL_STREAM_NAME)]
    t = []
    y = []
    for stamp, marker in zip(marker_stream["time_stamps"], marker_stream["time_series"]):
        if marker.startswith(SPELLER_MARKER_TRIAL_START):
            t.append(stamp)
        try:
            marker = json.loads(marker[0])
            if isinstance(marker, dict) and "target" in marker:
                y.append(int(marker["target"]))
        except:
            continue
    t = np.array(t)
    y = np.array(y)
    logger.debug(f"Found {t.size} markers and {y.size} labels")

    # Slice data to trials
    X = np.zeros((t.size, N_CHANNELS, int(TRIAL_TIME * sfreq)))
    for i_trial in range(t.size):
        idx = np.argmin(np.abs(time - t[i_trial]))
        X[i_trial, :, :] = data[idx:idx + int(TRIAL_TIME * sfreq)]
    logger.debug(f"The training data is of shape {X.shape} before resample")

    # Resample
    X = resample(X, int(TRIAL_TIME * FS), axis=2)
    logger.debug(f"The training data is of shape {X.shape} after resample")

    # Load codes
    V = np.repeat(np.load(Path(CODE_DIR) / CODE_FILE)["codes"], int(FS / PR), axis=1)
    logger.debug(f"Codes loaded from {Path(CODE_DIR) / CODE_FILE}")
    logger.debug(f"The codes are of shape {V.shape}")

    # Setup classifier
    rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event=EVENT, onset_event=ONSET_EVENT,
                                    encoding_length=ENCODING_LENGTH)
    stop = pyntbci.stopping.MarginStopping(estimator=rcca, segment_time=SEGMENT_TIME, fs=FS, target_p=TARGETED_ACCURACY)

    # Cross-validation
    folds = np.repeat(np.arange(N_FOLDS), int(X.shape[0] / N_FOLDS))
    accuracy = np.zeros(N_FOLDS)
    duration = np.zeros(N_FOLDS)
    for i_fold in range(N_FOLDS):

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
            for i_segment in range(int(X.shape[2] / (SEGMENT_TIME * FS))):

                # Apply classifier
                label = stop.predict(X_i[:, :, :int((1 + i_segment) * SEGMENT_TIME * FS)])[0]

                # Stop the trial if classified
                if label >= 0:
                    yh_tst[i_trial] = label
                    yh_dur[i_trial] = (1 + i_segment) * SEGMENT_TIME
                    break

        # Compute performance
        accuracy[i_fold] = np.mean(yh_tst == y_tst)
        duration[i_fold] = np.mean(yh_dur)
    logger.info(f"Cross-validated accuracy of {np.mean(accuracy):.3f} +/- {np.std(accuracy):.3f}")
    logger.info(f"Cross-validated duration of {np.mean(duration):.2f} +/- {np.std(duration):.2f}")

    # Fit classifier on all data
    rcca.fit(X, y)

    # Save classifier
    joblib.save(rcca, Path(CLASSIFIER_DIR) / CLASSIFIER_FILE)
    logger.debug(f"Classifier saved to {Path(CLASSIFIER_DIR) / CLASSIFIER_FILE}")

    return 0
