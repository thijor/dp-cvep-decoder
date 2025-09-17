import json
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import pylsl
import toml
from dareplane_utils.general.time import sleep_s
from dareplane_utils.logging.logger import get_logger
from dareplane_utils.signal_processing.filtering import FilterBank
from dareplane_utils.stream_watcher.lsl_stream_watcher import StreamWatcher
from fire import Fire
from numpy.typing import NDArray
from scipy.signal import resample

from cvep_decoder.utils.logging import logger


class OnlineDecoder:
    """Decoder class to evaluate a classifier based on data from an LSL stream.

    Parameters
    ----------
    decoder_file : Path
        Path to the classifier object to use for decoding.

    decoder_meta_file : Path
        Path to the dictionary with metadata about the classifier.

    marker_stream_name : str
        Name of the marker stream to read the start_eval_marker from which decoding will be started.

    data_stream_name : str
        Name of the input stream containing the signal used for decoding.

    decoder_stream_name : str
        Name of the output stream to which the result of classifier.predict() are written.

    buffer_size_s : float
        Defines the size of the buffer for the data and marker stream.

    start_eval_marker : str
        The marker that starts the continuous evaluation.

    max_eval_time_s : float
        The maximum time to try decoding a trial. Default is 10s.

    t_sleep_s : float
        Time to sleep between updates, defines the update frequency. Default is 0.1s.

    selected_channels : list[str] | list[int] | None
        If a list of channel names is provided, data of only those channels will be processed. Default is None, which
         means all channels are considered. If a list of integers is provided, they are interpreted as indices.
    """

    def __init__(
            self,
            decoder_file: Path,
            decoder_meta_file: Path,
            marker_stream_name: str,
            data_stream_name: str,
            decoder_stream_name: str,
            buffer_size_s: float,
            start_eval_marker: str,
            max_eval_time_s: float = 10,
            t_sleep_s: float = 0.1,
            selected_channels: list[str] | None = None,
    ):

        self.classifier_path = decoder_file
        self.classifier_meta_path = decoder_meta_file
        self.marker_stream_name = marker_stream_name
        self.data_stream_name = data_stream_name
        self.decoder_stream_name = decoder_stream_name
        self.buffer_size_s = buffer_size_s
        self.start_eval_marker = start_eval_marker
        self.max_eval_time_s = max_eval_time_s
        self.t_sleep_s = t_sleep_s
        self.selected_channels = selected_channels

        self.selected_ch_idx = None
        self.is_decoding: bool = False
        self.start_eval_time: float = 0.0
        self.internal_decoding_start_time: float = time.time()
        self.classifier = None
        self.classifier_meta = None

        self.input_mrk_sw: StreamWatcher | None = None
        self.input_sw: StreamWatcher | None = None
        self.input_sfreq: int | None = None
        self.input_chs_info: list[dict[str, str]] | None = None
        self.filterbank: FilterBank | None = None
        self.output_sw: StreamWatcher | None = None
        self.band = None
        self.classifier_input_sfreq = None

    # -------- Connection and initialization methods --------------------------
    def load_model(
            self,
            classifier_path: Path | None = None,
            classifier_meta_path: Path | None = None,
    ) -> int:
        """Loading the model and allowing for overwrites"""

        cp = (
            classifier_path
            if classifier_path is not None
            else self.classifier_path
        )
        cmp = (
            classifier_meta_path
            if classifier_meta_path is not None
            else self.classifier_meta_path
        )
        logger.info(f"Loading classifier from {cp=} and {cmp=}.")

        try:
            self.classifier = joblib.load(cp)
            self.classifier_meta = json.load(open(cmp, "r"))
        except FileNotFoundError:
            logger.error(f"Could not load classifier from {cp=} or {cmp=}. Validate that both exist.")
            return 1

        self.classifier_input_sfreq = self.classifier_meta["sfreq"]
        self.band = self.classifier_meta["fband"]

        return 0

    def connect_marker_stream(self):
        logger.info(f'Connecting to marker stream "{self.marker_stream_name}".')
        self.input_mrk_sw = StreamWatcher(self.marker_stream_name, buffer_size_s=self.buffer_size_s, logger=logger)
        self.input_mrk_sw.connect_to_stream()

        if len(self.input_mrk_sw.channel_names) != 1:
            logger.error("The marker stream should have exactly one channel.")
            return 1

        return 0

    def connect_data_stream(self):
        logger.info(f'Connecting to data stream "{self.data_stream_name}".')
        self.input_sw = StreamWatcher(self.data_stream_name, buffer_size_s=self.buffer_size_s, logger=logger)
        self.input_sw.connect_to_stream()

        self.input_sfreq = int(self.input_sw.inlet.info().nominal_srate())
        self.input_chs_info = [dict(ch_name=ch_name, type="EEG") for ch_name in self.input_sw.channel_names]

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
                raise logger.error(f"{self.selected_channels=} must be a list of `str` or `int` or `None`.")
            return 1

        return 0

    def create_filterbank(self):
        logger.info("Creating the classifier and the filter bank.")
        assert self.input_chs_info, (
            f"self.input_chs_info is {self.input_chs_info}. Please connect to "
            "the input lsl stream to derive channel information by calling "
            " `self.connect_data_stream()`"
        )

        self.filterbank = FilterBank(
            bands={"band": self.band},
            sfreq=self.input_sfreq,
            output="signal",
            n_in_channels=len(self.selected_ch_idx),
            filter_buffer_s=self.buffer_size_s,
        )

    def create_decoder_stream(self):
        logger.info(f'Creating the decoder stream "{self.decoder_stream_name}"')
        info = pylsl.StreamInfo(
            self.decoder_stream_name,
            type="MISC",
            channel_count=1,
            nominal_srate=pylsl.IRREGULAR_RATE,
            channel_format=pylsl.cf_int8,
            source_id="decoder_stream_id",
        )
        self.output_sw = pylsl.StreamOutlet(info)

    def init_all(self) -> int:
        """Return an int as this is exposed as a PCOMM `CONNECT_DECODER`"""
        self.connect_marker_stream()
        self.connect_data_stream()
        self.create_filterbank()
        self.create_decoder_stream()
        return 0

    # ------------------ Online functionality ---------------------------------

    def run(self) -> tuple[threading.Thread, threading.Event]:
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._run_loop,
            kwargs={"stop_event": stop_event},
        )
        thread.start()
        logger.debug("Started the run loop")
        return thread, stop_event

    def update(self):
        logger.debug(f"Updating the decoder - {self.is_decoding=}")

        self.input_sw.update()
        self.input_mrk_sw.update()
        self._filter()

        # check if decoding should start
        if not self.is_decoding:
            self.check_if_decoding_should_start()
        else:
            # Stop decoding on time-out
            if time.time() - self.internal_decoding_start_time > self.max_eval_time_s:
                logger.info(f"Stopping decoding after {self.max_eval_time_s=}")
                self.is_decoding = False
            else:
                # Decoding
                if self.is_decoding:
                    x = self._create_epoch()
                    if x.shape[2] > 0:
                        xs = self._resample(x)
                        self._classify(xs)

    def check_if_decoding_should_start(self):
        if self.input_mrk_sw is None:
            logger.error("No marker stream connected, cannot start decoding based on markers.")

        if self.input_mrk_sw.n_new > 0:
            markers = self.input_mrk_sw.unfold_buffer()[-self.input_mrk_sw.n_new:, 0]
            markers_t = self.input_mrk_sw.unfold_buffer_t()[-self.input_mrk_sw.n_new:]

            if self.start_eval_marker in markers:
                logger.debug(f"Starting decoding based on marker '{self.start_eval_marker}'")
                self.is_decoding = True
                self.input_mrk_sw.n_new = 0
                self.internal_decoding_start_time = time.time()

                # get time stamp of start_eval_marker --> consider inputs for epoch from this onwards
                idx = np.where(markers == self.start_eval_marker)[0]
                self.start_eval_time = markers_t[idx]

    def _filter(self):
        if self.input_sw is None:
            logger.error("No input stream connected, cannot filter input.")

        if self.input_sw.n_new > 0:
            x = self.input_sw.unfold_buffer()[-self.input_sw.n_new:, self.selected_ch_idx]
            t = self.input_sw.unfold_buffer_t()[-self.input_sw.n_new:]
            self.filterbank.filter(x, t)
            self.input_sw.n_new = 0  # makes sure samples are filtered only once

    def _create_epoch(self) -> NDArray:
        x = self.filterbank.get_data()[:, :, 0]
        t = self.input_sw.unfold_buffer_t()

        # Select only samples from start_eval_marker onwards
        idx = np.argmin(np.abs(t - self.start_eval_time))
        x = x[idx:, :]

        x = x.T[None, :, :]  # (1, n_channels, n_samples)
        logger.debug(f"Created epoch of shape {x.shape} (n_trials x n_channels x n_samples).")

        if np.isnan(x).sum() > 0:
            logger.error("NaNs found after epoching.")

        return x

    def _resample(self, x: NDArray) -> NDArray:

        if self.classifier_input_sfreq is not None:
            logger.debug(f"The data x is of shape {x.shape} (n_trials x n_channels x n_samples) before resample.")
            x = resample(
                x,
                num=int(x.shape[2] / self.input_sfreq * self.classifier_input_sfreq),
                axis=2,
            )
            logger.debug(f"The data x is of shape {x.shape} (n_trials x n_channels x n_samples) after resample.")

            if np.isnan(x).sum() > 0:
                logger.error("NaNs found after resampling")

        return x

    def _classify(self, x: NDArray):  # (n_trials, n_channels, n_samples) N.B. n_trials=1

        if x.shape[2] < int(self.t_sleep_s * self.classifier_input_sfreq):
            logger.debug(f"Classifying skipped as insufficient data: {x.shape[2]=}.")
            y = -1
        else:
            logger.debug(f"Classifying epoch of shape {x.shape} (n_trials x n_channels x n_samples).")
            y = self.classifier.predict(x)[0]
            logger.debug(f"Classified with prediction {y}.")

        # If y=-1 then the classifier is not yet sufficiently certain to emit the classification
        if y >= 0:
            logger.debug(f"Pushing prediction {y}.")
            self.output_sw.push_sample([np.int64(y)])
            self.is_decoding = False

    def _run_loop(self, stop_event: threading.Event):

        logger.debug("Starting the run loop")
        if self.input_sw is None or self.output_sw is None or self.classifier is None:
            logger.error("Streams or decoding not initialized, call init_all first.")

        while not stop_event.is_set():
            t_start = pylsl.local_clock()
            self.update()
            t_end = pylsl.local_clock()

            # reduce sleep by processing time
            dt_sleep = self.t_sleep_s - (t_end - t_start)
            sleep_s(dt_sleep)


def online_decoder_factory(
        config_path: Path = Path("./configs/decoder.toml"), preload: bool = True
):
    """Factory function to create an OnlineDecoder object from a config file."""
    cfg = toml.load(config_path)

    online_dec = OnlineDecoder(
        decoder_file=cfg["decoder"]["decoder_file"],
        decoder_meta_file=cfg["decoder"]["decoder_meta_file"],
        marker_stream_name=cfg["streams"]["marker_stream_name"],
        data_stream_name=cfg["streams"]["data_stream_name"],
        decoder_stream_name=cfg["streams"]["decoder_stream_name"],
        buffer_size_s=cfg["streams"]["buffer_size_s"],
        start_eval_marker=cfg["stimulus"]["trial_marker"],
        max_eval_time_s=cfg["online"]["max_eval_time_s"],
        selected_channels=cfg["data"].get("selected_channels", None),
    )

    if preload:
        online_dec.load_model()

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
