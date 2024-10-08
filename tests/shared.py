import json
import time
from pathlib import Path

import joblib
import numpy as np
import pytest
import toml
from pyntbci.classifiers import rCCA


@pytest.fixture(scope="session")
def provide_joblib_model():
    fs = 50
    seq_lengths = np.random.randint(2, 10, 10)
    seq = np.hstack([[i % 2] * seq_l for i, seq_l in enumerate(seq_lengths)])

    # just a simple two class example
    stimulus = np.vstack([seq, np.roll(seq, 4)])

    file = Path("./test.joblib")
    rcca = rCCA(stimulus, fs)

    # mock fit to use the model
    fitted = False
    while not fitted:
        try:
            x = np.ones((100, 5, 500)) + np.random.randn(100, 5, 500) * 0.1
            y = np.random.choice([0, 1], 100)
            rcca.fit(x, y)
            fitted = True
        except np.linalg.LinAlgError as e:
            print(
                "Encountered LinAlgError during fit, retrying with different random data"
            )
            # TODO: properly fix this to ensure the fit never encounters singular values
            raise ValueError(
                "Singular value in random sample data, please rerun pytest"
            )

    joblib.dump(rcca, file)

    meta_file = Path("./test_meta.json")
    meta = {"sfreq": fs, "fband": [1, 40]}
    json.dump(meta, open(meta_file, "w"))

    yield file

    for f in [file, meta_file]:
        if f.exists():
            f.unlink()


@pytest.fixture(scope="session")
def provide_config_toml():
    toml_dict = {
        "cvep": {"code_file": "./"},
        "training": {
            "data_root": "./data/",
            "training_files_glob": "sub-P001_*.xdf",
            "out_file": "./test.joblib",
            "out_file_meta": "./test_meta.json",
            "features": {"target_freq_hz": 120, "passband_hz": [1, 40]},
        },
        "online": {
            "sleep_s": 0.1,
            "input": {
                "lsl_stream_name": "test_eeg_data",
                "buffer_size_s": 3.0,
                "lsl_marker_stream_name": "test_marker_stream",
                "selected_channels": [1, 2],
            },
            "output": {"lsl_stream_name": "test_cvep_decoder", "buffer_size_s": 1.0},
            "classifier": {"file": "./test.joblib", "meta_file": "./test_meta.json"},
            "eval": {
                "eval_after_type": "time",
                "eval_after_s": 0.1,
                "eval_after_nsamples": 10,
                "start": {"marker": "1", "max_time_s": 3, "pre_eval_start_s": 0.0},
                "marker": {"trigger_marker": ["2", "3"]},
            },
            "early_stop": {},
        },
    }

    file = Path("./decoder_config.toml")
    toml.dump(toml_dict, open(file, "w"))

    yield file

    if file.exists():
        file.unlink()
