from pathlib import Path

import joblib
import numpy as np
import pytest
import toml
from pyntbci.classifiers import rCCA


@pytest.fixture
def provide_joblib_model():
    fs = 50
    seq_lengths = np.random.randint(2, 10, 10)
    seq = np.hstack([[i % 2] * seq_l for i, seq_l in enumerate(seq_lengths)])

    # just a simple two class example
    stimulus = np.vstack([seq, np.roll(seq, 4)])

    file = Path("./test.joblib")
    rcca = rCCA(stimulus, fs)
    joblib.dump(rcca, file)

    yield file

    file.unlink()


@pytest.fixture
def provide_config_toml():
    toml_dict = {
        "cvep": {"code_file": "./"},
        "training": {
            "data_root": "./data/",
            "training_files_glob": "sub-P001_*.xdf",
            "out_file": "./test.joblib",
            "features": {"target_freq_hz": 6, "passband_hz": [1, 40]},
        },
        "online": {
            "sleep_s": 0.1,
            "input": {"lsl_stream_name": "test_eeg_data", "buffer_size_s": 1.0},
            "output": {"lsl_stream_name": "test_cvep_decoder", "buffer_size_s": 1.0},
            "classifier": {"file": "./test.joblib", "band": [1, 40]},
            "eval": {
                "eval_after_type": "time",
                "eval_after_s": 0.1,
                "eval_after_nsamples": 10,
                "marker": {
                    "stream_name": "my_marker_stream",
                    "trigger_marker": [1, 2, 3],
                },
            },
            "early_stop": {},
        },
    }

    file = Path("./decoder_config.toml")
    toml.dump(toml_dict, open(file, "w"))

    yield file

    file.unlink()
