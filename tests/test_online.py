# Testing online functionality
import logging
import threading
import time
from pathlib import Path

import numpy as np
import pylsl
import pytest
from dareplane_utils.logging.logger import get_logger

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
def spawn_lsl_data_stream() -> threading.Event:

    stop_event = threading.Event()
    stop_event.clear()
    th = threading.Thread(target=provide_lsl_stream, args=(stop_event,))
    th.start()

    yield stop_event

    # teardown
    stop_event.set()
    th.join()


@pytest.fixture(scope="session")  # only create once for all tests
def provide_marker_stream() -> pylsl.StreamOutlet:
    outlet = pylsl.StreamOutlet(
        pylsl.StreamInfo(
            "test_marker_stream",
            "MISC",
            1,
            pylsl.IRREGULAR_RATE,
            "string",
            "test_marker_data_id",
        )
    )

    return outlet


# should not be on session scope, as tests should work on a new instance
@pytest.fixture
def provide_cvep_decoder(
    provide_joblib_model: Path, provide_config_toml: Path
) -> OnlineDecoder:

    online_decoder = online_decoder_factory(provide_config_toml)

    return online_decoder


@pytest.fixture
def provide_running_decoder(
    spawn_lsl_data_stream: threading.Event,
    provide_cvep_decoder: OnlineDecoder,
    provide_marker_stream: pylsl.StreamOutlet,
) -> tuple[OnlineDecoder, pylsl.StreamOutlet]:
    cvd = provide_cvep_decoder
    cvd.selected_channels = None  # select all channels for this use case
    outlet = provide_marker_stream

    cvd.init_all()
    cvd.eval_after_s = 0.1

    thread, stop_ev = cvd.run()

    yield cvd, outlet

    stop_ev.set()
    thread.join()


def test_connection_to_stream(
    spawn_lsl_data_stream, provide_cvep_decoder, provide_marker_stream
):
    stop_ev = spawn_lsl_data_stream
    cvd = provide_cvep_decoder

    cvd.connect_input_streams()


def test_outlet_creation(spawn_lsl_data_stream, provide_cvep_decoder):

    cvd = provide_cvep_decoder
    cvd.create_output_stream()

    streams = pylsl.resolve_byprop("name", "test_cvep_decoder", timeout=1)
    assert len(streams) == 1


def test_filterbank_init(
    spawn_lsl_data_stream, provide_cvep_decoder, provide_marker_stream
):

    cvd = provide_cvep_decoder

    # error as we are not connected yet
    with pytest.raises(AssertionError):
        cvd.create_filterbank()

    cvd.connect_input_streams()

    # now channel info should be there and creation should work
    cvd.create_filterbank()

    assert cvd.filterbank is not None
    assert cvd.filterbank.n_in_channels == 2
    assert cvd.filterbank.bands["band"] == [1, 40]
    assert len(cvd.filterbank.bands) == 1  # only a single band for the test cfg


def test_start_decoding_by_marker(
    spawn_lsl_data_stream, provide_cvep_decoder, provide_marker_stream
):
    cvd = provide_cvep_decoder

    outlet = provide_marker_stream

    cvd.init_all()

    # starting should also reset the n_new of the buffers -> raise then to check
    cvd.input_sw.n_new = 10
    cvd.filterbank.n_new = 10

    assert cvd.is_decoding is False

    outlet.push_sample(["1"])
    time.sleep(0.01)
    cvd.update()

    assert cvd.is_decoding is True
    assert (
        cvd.input_sw.n_new == cvd.pre_eval_start_n
    )  # should be zero according to test config
    assert cvd.filterbank.n_new == cvd.pre_eval_start_n
    assert cvd.input_mrk_sw.n_new == 0


def test_stop_decoding_after_time_expired(
    spawn_lsl_data_stream, provide_cvep_decoder, provide_marker_stream
):
    cvd = provide_cvep_decoder

    outlet = provide_marker_stream
    cvd.init_all()

    cvd.max_eval_time_s = 0.3

    outlet.push_sample(["1"])
    time.sleep(0.01)
    cvd.update()
    assert cvd.is_decoding is True
    time.sleep(0.5)
    cvd.update()
    assert cvd.is_decoding is False


@pytest.mark.parametrize("eval_after_type", ["time", "nsamples", "markers"])
def test_classification_triggers(
    eval_after_type, spawn_lsl_data_stream, provide_cvep_decoder, provide_marker_stream
):
    cvd = provide_cvep_decoder
    outlet = provide_marker_stream
    cvd.init_all()

    cvd.eval_after_type = eval_after_type
    cvd.eval_after_nsamples = 10
    cvd.eval_after_s = 0.1

    # start decoding
    outlet.push_sample(["1"])
    time.sleep(0.01)
    cvd.update()

    match eval_after_type:
        case "nsamples":
            # not enough yet
            assert not cvd.enough_data_for_next_prediction() and cvd.is_decoding
            # should be enough
            time.sleep(0.2)
            cvd.input_sw.update()

            assert cvd.enough_data_for_next_prediction() and cvd.is_decoding

        case "time":
            # -> not enough
            assert not cvd.enough_data_for_next_prediction() and cvd.is_decoding
            time.sleep(0.1)
            # now enough
            cvd.input_sw.update()
            assert cvd.enough_data_for_next_prediction() and cvd.is_decoding

        case "marker":
            outlet.push_sample(["5"])  # should not trigger
            time.sleep(0.01)
            cvd.input_mrk_sw.update()
            assert not cvd.enough_data_for_next_prediction() and cvd.is_decoding

            outlet.push_sample(["2"])  # should trigger
            time.sleep(0.01)
            cvd.input_mrk_sw.update()
            assert cvd.enough_data_for_next_prediction() and cvd.is_decoding


def test_epoch_creation(
    spawn_lsl_data_stream, provide_cvep_decoder, provide_marker_stream
):
    cvd = provide_cvep_decoder
    outlet = provide_marker_stream
    cvd.init_all()
    cvd.eval_after_s = 1  # we just want to manually trigger

    # collect some pre data
    time.sleep(0.1)
    cvd.update()

    # start decoding
    outlet.push_sample(["1"])
    time.sleep(0.01)
    cvd.update()

    # more data
    time.sleep(0.1)
    cvd.update()

    cvd._filter()  # to get data into the filterbank buffer
    x = cvd._create_epoch()
    assert x.ndim == 3
    assert len(x) == 1
    assert x.shape[1] == len(cvd.selected_ch_idx)


def test_resampling(spawn_lsl_data_stream, provide_cvep_decoder, provide_marker_stream):

    cvd = provide_cvep_decoder
    outlet = provide_marker_stream
    cvd.init_all()

    # collect some data
    time.sleep(0.3)
    cvd.update()
    cvd._filter()  # to get data into the filterbank buffer
    x = cvd._create_epoch()
    xs = cvd._resample(x)

    # df = pl.concat(
    #     [
    #         pl.DataFrame(
    #             {"x": x[0, 0, :].T, "time": np.linspace(0, 1, x.shape[2]), "src": "raw"}
    #         ),
    #         pl.DataFrame(
    #             {
    #                 "x": xs[0, 0, :].T,
    #                 "time": np.linspace(0, 1, xs.shape[2]),
    #                 "src": "resampled",
    #             }
    #         ),
    #     ]
    # )
    #
    # px.scatter(df, x="time", y="x", color="src").show()

    # the example metadata we use specifies 50Hz -> we should have half
    # of what the input was. Be lenient in rtol
    assert np.allclose(x.shape[2] / xs.shape[2], 2, rtol=0.05)


def test_running_loop(provide_running_decoder):
    cvd, outlet = provide_running_decoder

    # read from the output stream
    inlet = pylsl.StreamInlet(
        pylsl.resolve_byprop("name", cvd.output_stream_name, timeout=1)[0]
    )
    samples, sample_times = inlet.pull_chunk()

    # nothing there as we did not start decoding
    assert samples == []
    assert sample_times == []

    # start epoch
    outlet.push_sample(["1"])
    time.sleep(0.5)

    # Now we should have decoder output
    samples, sample_times = inlet.pull_chunk()

    assert len(samples) > 0

    # also assert that the values are as expected
    assert samples[0][0] == 0 or samples[0][0] == 1
