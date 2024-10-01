import json
import threading
import time
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pylsl
import toml
from dareplane_utils.general.time import sleep_s
from dareplane_utils.logging.logger import get_logger
from dareplane_utils.signal_processing.filtering import FilterBank
from dareplane_utils.stream_watcher.lsl_stream_watcher import StreamWatcher
from fire import Fire
from mne.filter import resample
from numpy.typing import NDArray
from pyntbci.classifiers import rCCA
from pyntbci.stopping import MarginStopping

from cvep_decoder.utils.logging import logger


class OnlineDecoder:
    """Decoder class to evaluate a classifier based on data from a LSL stream.

    Parameters
    ----------
    classifier : rcca
        classifier object to use for decoding. Either a `rCCA` or a `MarginStopping` object from pyntbci.
        The MarginStopping is used in combination with the dp-cvep-speller

    classifier_meta: dict
        Dictionary with metadata about the classifier. Should contain:
            - sfreq: float
            - band: tuple[float, float]

    input_stream_name : str
        Name of the input stream (containing the signal).

    output_stream_name : str
        Name of the output stream for the result of classifier.predict().

    input_buffer_size_s : float
        Temporal size of the window that should be passed to the model. Will
        define the size of the buffer for the `input` StreamWatcher and the
        buffer size of the FilterBank.

    t_sleep_s : float
        Time to sleep between updates, defines the update frequency. Default is 0.1s.

    start_eval_marker : str
        the marker that starts the continuous evaluation

    max_eval_time_s : float
        maximum time to look to try decoding a trial. Default is 10s.

    marker_stream_name : str | None
        Optional name of the marker stream. If a stream is provided, the projections will be synchronised
        with the markers. Else,....fixed lenght epochs? Default is None.

    decode_trigger_markers : list[str]
        List of markers to trigger a decoder evaluation. Only used if marker_stream_name is not None. Default is None.

    offset_nsamples : float
        Offset of the epochs wrt the markers. Only used if marker_stream_name is not None. Default is 0.

    eval_after_type : Literal["time", "samples", "markers"]
        Type of evaluation trigger. Default is "time". This will result in an
        evaluation of the classifier every `eval_after_s` seconds. If "samples"
        is chosen, the classifier evaluates after `eval_after_nsamples` new
        samples arrived at the input StreamWatchers.

    eval_after_s : float
        Time after which the classifier should be evaluated. Default is 1s.

    eval_after_nsamples : int
        Number of samples after which the classifier should be evaluated.
        Default is 10.

    pre_eval_start_s : float
        Time window of data to include before the start marker for classification arrived. Default is 0.0s.

    selected_channels : list[str] | list[int] | None
        if a list of channel names is provided, data of only those channels will
        be processed. Default is None -> all channels are considered. If list of integers is provided, they are interpreted as indeces.

    """

    def __init__(
        self,
        classifier: rCCA | ,
        classifier_meta: dict,
        input_stream_name: str,
        marker_stream_name: str,
        output_stream_name: str,
        input_buffer_size_s: float,
        start_eval_marker: str,
        max_eval_time_s: float = 10,
        classifier_input_sfreq: float | None = None,
        t_sleep_s: float = 0.1,
        decode_trigger_markers: list[str] | None = None,
        offset_nsamples: float = 0,
        eval_after_type: Literal["time", "nsamples", "markers"] = "time",
        eval_after_s: float = 1,
        eval_after_nsamples: int = 10,
        pre_eval_start_s: float = 0.0,
        selected_channels: list[str] | None = None,
    ):
        self.classifier = classifier
        self.classifier_meta = classifier_meta
        self.input_stream_name = input_stream_name
        self.output_stream_name = output_stream_name
        self.buffer_size_s = input_buffer_size_s
        self.max_eval_time_s = max_eval_time_s
        self.t_sleep_s = t_sleep_s
        self.marker_stream_name = marker_stream_name
        self.decode_trigger_markers = decode_trigger_markers
        self.offset_nsamples = offset_nsamples
        self.curr_offset_nsamples = (
            offset_nsamples  # used of epochs are sliced by markers
        )

        self.start_eval_marker = start_eval_marker
        self.pre_eval_start_s = pre_eval_start_s
        self.eval_after_s = eval_after_s
        self.eval_after_nsamples = eval_after_nsamples

        self.selected_channels = selected_channels
        self.selected_ch_idx = None  # will be set once connected to the input stream

        self.is_decoding: bool = False
        self.start_eval_time: float = 0.0

        self.input_sw: StreamWatcher = None
        # Will be derived once the input_sw is connected
        self.input_sfreq: float = None
        self.input_chs_info: list[dict[str, str]] = None
        self.input_mrk_sw: StreamWatcher = None

        self.filterbank: FilterBank = None
        self.output_sw: StreamWatcher = None

        # Derived attributes
        self.classifier_input_sfreq = self.classifier_meta["sfreq"]
        self.band = self.classifier_meta["band"]
        self.pre_eval_start_n = None  # will be set once connected to input stream

        eval_type_map = {
            "time": self.check_enough_time_based,
            "nsamples": self.check_enough_sample_based,
            "markers": self.check_enough_marker_based,
        }

        self.enough_data_for_next_prediction = eval_type_map[eval_after_type]

        self.classify = self._classify_early_stopping if isinstance(classifier, MarginStopping) else self._classify

    # -------- Connection and initialization methods --------------------------
    def connect_input_streams(self):
        logger.info(f'Connecting to input stream "{self.input_stream_name}"')
        self.input_sw = StreamWatcher(
            self.input_stream_name, buffer_size_s=self.buffer_size_s, logger=logger
        )

        self.input_sw.connect_to_stream()

        # set derived properties
        self.input_sfreq = self.input_sw.inlet.info().nominal_srate()
        self.pre_eval_start_n = int(self.pre_eval_start_s * self.input_sfreq)
        self.input_chs_info = [
            dict(ch_name=ch_name, type="EEG") for ch_name in self.input_sw.channel_names
        ]

        if self.selected_channels is None:
            self.selected_ch_idx = list(range(len(self.input_chs_info)))
        else:
            if isinstance(self.selected_channels[0], str):
                self.selected_ch_idx = [
                    self.input_sw.channel_names.index(ch)
                    for ch in self.selected_channels
                ]
            elif isinstance(self.selected_channels[0], int):
                self.selected_ch_idx = self.selected_channels
            else:
                raise ValueError(
                    f"{self.selected_channels=} must be a list of `str` or `int` or `None`."
                )

        # We require the marker stream to start the decoding (start trials)
        logger.debug(f'Connecting to marker stream "{self.marker_stream_name}"')
        self.input_mrk_sw = StreamWatcher(
            self.marker_stream_name, buffer_size_s=self.buffer_size_s, logger=logger
        )
        self.input_mrk_sw.connect_to_stream()
        if len(self.input_mrk_sw.channel_names) != 1:
            logger.error("The marker stream should have exactly one channel")

    def create_filterbank(self):
        logger.debug("Creating the classifier and the filter bank.")
        assert self.input_chs_info, (
            f"self.input_chs_info is {self.input_chs_info}. Please connect to "
            "the input lsl stream to derive channel information by calling "
            " `self.connect_input_streams()`"
        )

        self.filterbank = FilterBank(
            bands={"band": self.band},
            sfreq=self.input_sfreq,
            output="signal",
            n_in_channels=len(self.selected_ch_idx),
            filter_buffer_s=self.buffer_size_s,
        )

    def create_output_stream(self):
        logger.info(f'Creating the output stream "{self.output_stream_name}"')
        info = pylsl.StreamInfo(
            self.output_stream_name,
            "MISC",
            channel_count=1,
            nominal_srate=pylsl.IRREGULAR_RATE,
            channel_format="int32",
            source_id="output_stream_id",
        )
        self.output_sw = pylsl.StreamOutlet(info)

    def init_all(self) -> int:
        """Return an int as this is exposed as a PCOMM `CONNECT_DECODER`"""
        self.connect_input_streams()
        self.create_filterbank()
        self.create_output_stream()
        return 0

    # ------------------ Online functionality ---------------------------------

    def check_enough_time_based(self) -> bool:
        if (self.eval_after_s * self.input_sfreq) <= self.input_sw.n_new:
            return True
        else:
            return False

    def check_enough_sample_based(self) -> bool:
        if self.eval_after_nsamples <= self.input_sw.n_new:
            return True
        else:
            return False

    def check_enough_marker_based(self) -> bool:
        n_new = self.input_mrk_sw.n_new
        if n_new == 0:
            return False

        markers = self.input_mrk_sw.unfold_buffer()[-n_new:, 0]
        markers_t = self.input_mrk_sw.unfold_buffer_t()[-n_new:]

        trigger_marker_indices = [
            i for i, m in enumerate(markers) if m in self.decode_trigger_markers
        ]

        if any(trigger_marker_indices):

            # find the closest match of time points between the last trigger marker and data sample times
            t = self.input_sw.unfold_buffer_t()[-self.filterbank.n_new :]
            idx_end = np.abs(markers_t[trigger_marker_indices[-1], None] - t).argmin(
                axis=1
            )

            # the offset we will effectivefly use is the global shift (by self.offset_nsamples)
            # plus wherever the triggering marker is relative to the latest data
            # sample from the input_sw
            self.curr_offset_nsamples = self.offset_nsamples + (len(t) - idx_end)

            return True
        else:
            return False

    def _filter(self):
        if self.input_sw is None:
            logger.error(
                "Input StreamWatcher not connected, use `self.connect_input_streams` first"
            )

        if self.input_sw.n_new == 0:
            logger.debug("No new samples to filter")
        else:
            self.filterbank.filter(
                self.input_sw.unfold_buffer()[
                    -self.input_sw.n_new :, self.selected_ch_idx
                ],
                self.input_sw.unfold_buffer_t()[-self.input_sw.n_new :],
            )

            logger.debug(f"Added {self.input_sw.n_new=} samples")
            self.input_sw.n_new = 0  # reset to ensure only additional data gets filled onto the filterbank

    def _create_epoch(self) -> NDArray:
        """
        Always use all new data in the filter buffer triggering this methods
        is timed to the desired evaluation trigger frequency
        """

        x = self.filterbank.get_data()

        if x.shape[0] < self.filterbank.n_new:
            logger.warning(
                f"Buffer to small for providing all data for {self.filterbank.n_new=}."
                f" Will continue with only {len(x) - self.curr_offset_nsamples}."
                f" Current buffer size {self.buffer_size_s=}."
            )

        logger.debug(
            f"Creating one epoch from the {self.filterbank.n_new=} latest samples."
            f" Using {self.curr_offset_nsamples=} and {x.shape=}"
        )

        selection_slice = slice(
            len(x) - (self.filterbank.n_new + self.curr_offset_nsamples),
            len(x) - self.curr_offset_nsamples,
        )
        x = x[selection_slice, :, 0]
        x = x.T[None, :, :]  # (1, n_channel, n_times)

        if np.isnan(x).sum() > 0:
            logger.error("NaNs found after epoching")

        return x

    def _resample(self, x: NDArray) -> NDArray:

        if self.classifier_input_sfreq is not None:
            logger.debug(f"Resampling to {self.classifier_input_sfreq} Hz")
            down = self.input_sfreq / self.classifier_input_sfreq
            x = resample(x.astype("float64"), down=down, axis=-1).astype("float32")
            if np.isnan(x).sum() > 0:
                logger.error("NaNs found after resampling")

        return x

    def check_if_decoding_should_start(self):
        if self.input_mrk_sw is None:
            logger.error(
                "No marker stream connected, cannot start decoding based on markers"
            )

        if self.input_mrk_sw.n_new > 0:
            markers = self.input_mrk_sw.unfold_buffer()[-self.input_mrk_sw.n_new :, 0]
            markers_t = self.input_mrk_sw.unfold_buffer_t()[-self.input_mrk_sw.n_new :]

            logger.debug(f"Checking for {self.start_eval_marker=}")
            logger.debug(f"Checking through {markers=}, {markers_t}")

            if self.start_eval_marker in markers:
                logger.debug(
                    f"Starting decoding based on marker {self.start_eval_marker}"
                )
                self.is_decoding = True
                self.start_eval_time = time.time()

                # reset the buffers
                logger.debug(
                    f"Resetting buffers for decoding - using {self.pre_eval_start_n=}"
                )
                # Important note: only keep the lookback in the raw data,
                # the filterbank will be populated during the loop
                self.input_sw.n_new = self.pre_eval_start_n
                self.filterbank.n_new = 0
                self.input_mrk_sw.n_new = 0

    def update(self):

        self.input_sw.update()
        self.input_mrk_sw.update()

        # check if decoding should start
        if not self.is_decoding:
            self.check_if_decoding_should_start()

        else:
            if time.time() - self.start_eval_time > self.max_eval_time_s:
                logger.info("Stopping decoding after max_eval_time_s")
                self.is_decoding = False

            else:
                # Do the regular decoding if enough data received
                # Enough new data -> evaluate classifier
                if self.enough_data_for_next_prediction() and self.is_decoding:

                    self._filter()

                    x = self._create_epoch()

                    xs = self._resample(
                        x
                    )  # required to align with codes for rCCA classifier

                    self.classify(xs)

    # the plain classifier that provides a class for every iteration
    def _classify(self, x: NDArray):  # (batch_size, n_channel, n_times)

        logger.debug(f"Classifying for {x.shape=}")
        y = self.classifier.predict(x)[0]
        if np.isnan(y).sum() > 0:
            logger.error("NaNs found after embedding")

        logger.debug(f"Pushing prediction {y}")
        self.output_sw.push_sample([y])

    def _classify_early_stopping(self, x: NDArray):  # (batch_size, n_channel, n_times)

        logger.debug(f"Classifying for {x.shape=}")

        # This is assuming that the classifier is of type `pyntbci.stopping.MarginStopping`
        y = self.classifier.predict(x)[0]
        if np.isnan(y).sum() > 0:
            logger.error("NaNs found in prediction")

        if (
            y > 0
        ):  # this is how `pyntbci.stopping.MarginStopping` marks that it found a prediction
            # later we need to shift with -1 to go back to 0 indexing
            logger.debug(f"Pushing prediction {y}")

            # `speller_select` is the prefix expected by dp-cvep-speller
            self.output_sw.push_sample([f"speller_select {y-1}"])

    def _run_loop(self, stop_event: threading.Event):

        logger.debug("Starting the run loop")
        if self.input_sw is None or self.output_sw is None or self.classifier is None:
            logger.error("OnlineDecoder not initialized, call init_all first")

        while not stop_event.is_set():
            t_start = pylsl.local_clock()
            self.update()
            t_end = pylsl.local_clock()

            # reduce sleep by processing time
            dt_sleep = self.t_sleep_s - (t_end - t_start)
            # logger.debug(f"Sleeping for {dt_sleep=}")
            sleep_s(dt_sleep)
            # logger.debug("Woke up from sleep")

    def run(self) -> tuple[threading.Thread, threading.Event]:
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._run_loop,
            kwargs={"stop_event": stop_event},
        )
        thread.start()
        logger.debug("Started the run loop")
        return thread, stop_event


def online_decoder_factory(config_path: Path = Path("./configs/decoder.toml")):
    """Factory function to create an OnlineDecoder object from a config file."""
    cfg = toml.load(config_path)

    classifier = joblib.load(cfg["online"]["classifier"]["file"])
    classifier_meta = json.load(open(cfg["online"]["classifier"]["meta_file"], "r"))

    online_dec = OnlineDecoder(
        classifier,
        classifier_meta=classifier_meta,
        input_stream_name=cfg["online"]["input"]["lsl_stream_name"],
        output_stream_name=cfg["online"]["output"]["lsl_stream_name"],
        marker_stream_name=cfg["online"]["input"]["lsl_marker_stream_name"],
        input_buffer_size_s=cfg["online"]["input"]["buffer_size_s"],
        start_eval_marker=cfg["online"]["eval"]["start"]["marker"],
        eval_after_type=cfg["online"]["eval"]["eval_after_type"],
        eval_after_s=cfg["online"]["eval"].get("eval_after_s", 1),
        eval_after_nsamples=cfg["online"]["eval"].get("eval_after_nsamples", 1),
        pre_eval_start_s=cfg["online"]["eval"]["start"]["pre_eval_start_s"],
        max_eval_time_s=cfg["online"]["eval"]["start"]["max_time_s"],
        selected_channels=cfg["online"]["input"].get("selected_channels", None),
    )

    return online_dec


def cli_run_decoder(
    conf_pth: Path = Path("./configs/decoder.toml"), log_level: int = 30
):
    # if the CLI is run, we most likely also want a console output
    logger = get_logger("cvep_decoder", add_console_handler=True)
    logger.setLevel(log_level)

    logger.debug(f"Starting the decoder with {conf_pth=}")
    online_dec = online_decoder_factory(conf_pth)
    online_dec.init_all()
    thread, stop_event = online_dec.run()
    return thread, stop_event


if __name__ == "__main__":
    Fire(cli_run_decoder)
