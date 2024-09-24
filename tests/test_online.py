# Testing online funcionality
import logging
import os
import sys
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import pylsl
import pytest
from dareplane_utils.logging.logger import get_logger
from pyntbci.classifiers import rCCA

from cvep_decoder.online_decoding import OnlineDecoder, online_decoder_factory
from tests.shared import provide_config_toml, provide_joblib_model

# Overwrite the logger to have a console handler for unit test
logger = get_logger("cvep_decoder", add_console_handler=True)
logger.setLevel(logging.DEBUG)


def provide_lsl_stream(
    stop_event: threading.Event, srate: float = 100, nsamples: int = 1000
):
    # block print to stdout to suppress noise
    outlet = pylsl.StreamOutlet(
        pylsl.StreamInfo(
            "test_eeg_data", "EEG", 5, srate, "float32", "test_eeg_data_id"
        )
    )

    data = np.tile(np.linspace(0, 1, nsamples), (5, 1))
    data = data.T * np.arange(1, 6)  # 5 channels with linear increase
    data = data.astype(np.float32)

    isampl = 0
    nsent = 0
    tstart = time.time_ns()
    while not stop_event.is_set():
        dt = time.time_ns() - tstart
        req_samples = int((dt / 1e9) * srate) - nsent
        if req_samples > 0:
            outlet.push_chunk(data[isampl : isampl + req_samples, :].tolist())
            nsent += req_samples
            isampl = (isampl + req_samples) % data.shape[0]  # wrap around

        time.sleep(1 / srate)


@pytest.fixture(scope="session")  # only create once for all tests
def spawn_lsl_stream() -> threading.Event:

    stop_event = threading.Event()
    stop_event.clear()
    th = threading.Thread(target=provide_lsl_stream, args=(stop_event,))
    th.start()

    yield stop_event

    # teardown
    stop_event.set()
    th.join()


@pytest.fixture
def provide_cvep_decoder(
    provide_joblib_model: Path, provide_config_toml: Path
) -> OnlineDecoder:

    online_decoder = online_decoder_factory(provide_config_toml)

    return online_decoder


def test_connection_to_stream(spawn_lsl_stream, provide_cvep_decoder):
    stop_ev = spawn_lsl_stream
    cvd = provide_cvep_decoder

    cvd.connect_input_streams()


def test_outlet_creation(spawn_lsl_stream, provide_cvep_decoder):

    cvd = provide_cvep_decoder
    cvd.create_output_stream()

    streams = pylsl.resolve_byprop("name", "test_cvep_decoder", timeout=1)
    assert len(streams) == 1


def test_filterbank_init(spawn_lsl_stream, provide_cvep_decoder):

    cvd = provide_cvep_decoder

    # error as we are not connected yet
    with pytest.raises(AssertionError):
        cvd.create_filterbank()

    cvd.connect_input_streams()

    # now channel info should be there and creation should work
    cvd.create_filterbank()

    assert cvd.filterbank is not None
    assert cvd.filterbank.n_in_channels == 5
    assert cvd.filterbank.bands["band"] == [1, 40]
    assert len(cvd.filterbank.bands) == 1  # only a single band for the test cfg
