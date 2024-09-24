import threading
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pylsl
import toml
from dareplane_utils.general.time import sleep_s
from dareplane_utils.signal_processing.filtering import FilterBank
from dareplane_utils.stream_watcher.lsl_stream_watcher import StreamWatcher
from mne.filter import resample
from numpy.typing import NDArray
from pyntbci.classifiers import rCCA

from cvep_decoder.utils.logging import logger

"""
Adjustments in new version since 2024-09-24:
- the classifier file should contain info about:
    - expected sampling frequency for input
    - band to filter for in feature generation process

"""


class OnlineDecoder:
    """Decoder class to evaluate a classifier based on data from a LSL stream.

    Parameters
    ----------

    classifier : rcca
        classifier object to use for decoding

    input_stream_name : str
        Name of the input stream (containing the signal).

    output_stream_name : str
        Name of the output stream (containing the embeddings).

    input_window_seconds : float
        Temporal size of the window that should be passed to the  model.

    t_sleep_s : float
        Time to sleep between updates, defines the update frequency. Default is 0.1s.

    band : tuple[float, float]
        Band to filter the signal before passing it to the model. Default is (0.5, 40).

    marker_stream_name : str | None
        Optional name of the marker stream. If a stream is provided, the projections will be synchronised
        with the markers. Else,....fixed lenght epochs? Default is None.

    markers : list[str]
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

    """

    def __init__(
        self,
        classifier: rCCA,
        input_stream_name: str,
        output_stream_name: str,
        input_window_seconds: float,
        new_sfreq: float | None = None,
        band: tuple[float, float] = (0.5, 40),
        t_sleep_s: float = 0.1,
        marker_stream_name: str | None = None,
        markers: list[str] | None = None,
        offset_nsamples: float = 0,
        eval_after_type: Literal["time", "samples", "markers"] = "time",
        eval_after_s: float = 1,
        eval_after_nsamples: int = 10,
    ):
        self.classifier = classifier
        self.input_stream_name = input_stream_name
        self.output_stream_name = output_stream_name
        self.input_window_seconds = input_window_seconds
        self.band = band
        self.t_sleep_s = t_sleep_s
        self.marker_stream_name = marker_stream_name
        self.markers = markers
        self.offset_nsamples = offset_nsamples

        self.input_sw: StreamWatcher = None
        # Will be derived once the input_sw is connected
        self.input_sfreq: float = None
        self.input_chs_info: list[dict[str, str]] = None
        self.input_mrk_sw: StreamWatcher = None

        # have a list with relevant streamwatchers to avoid if statements in update
        # add mrk_sw if necessary
        self.input_sws: list[StreamWatcher] = [self.input_sw]

        self.filterbank: FilterBank = None
        self.output_sw: StreamWatcher = None

        # -------- Derived attributes ----------------------------------------
        self.buffer_size_s = 3 * (
            max(self.input_window_seconds, self.t_sleep_s) + abs(self.offset_nsamples)
        )

        self.model_sfreq = 1  # TODO: what is this?

        self.enough_data_for_next_prediction

    def n_outputs(self):
        x = np.zeros(
            (
                1,
                len(self.input_chs_info),
                int(self.input_window_seconds * self.model_sfreq),
            ),
            dtype=np.float32,
        )
        y = self.classifier.predict(x)
        return y.shape[1]

    # -------- Connection and initialization methods --------------------------
    def connect_input_streams(self):
        logger.info(f'Connecting to input stream "{self.input_stream_name}"')
        self.input_sw = StreamWatcher(
            self.input_stream_name, buffer_size_s=self.buffer_size_s, logger=logger
        )

        self.input_sw.connect_to_stream()

        # set input_sfreq and input_chs_info
        self.input_sfreq = self.input_sw.inlet.info().nominal_srate()
        self.input_chs_info = [
            dict(ch_name=ch_name, type="EEG") for ch_name in self.input_sw.channel_names
        ]

        if self.marker_stream_name:
            logger.info(f'Connecting to marker stream "{self.marker_stream_name}"')
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
            n_in_channels=len(self.input_chs_info),
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
            source_id="signal_embedding",  # TODO should it be more unique?
        )
        self.output_sw = pylsl.StreamOutlet(info)

    def init_all(self):
        self.connect_input_streams()
        self.create_filterbank()
        self.create_output_stream()

    # ------------------ Online functionality ---------------------------------

    def check_enough_time_based(self):
        return False

    def check_enough_sample_based(self):
        return False

    def check_enough_marker_based(self):
        return False

    def _create_epoch(self):
        """
        Always use all new data in the filter buffer triggering this methods
        is timed to the desired evaluation trigger frequency
        """

        x = self.filterbank.get_data()

        logger.debug(
            f"Creating one epoch from the {self.filterbank.n_new} latest samples"
        )
        x = x[-self.filterbank.n_new :, :, 0]
        x = x.T[None, :, :]  # (1, n_channel, n_times)

    def update(self):

        # Update all StreamWatchers
        for sw in self.input_sws:
            sw.update()

        # Enough new data -> evaluate classifier
        if self.enough_data_for_next_prediction():

            self._filter()

            # [ ] TODO: -- < resample where? >
            self._resample()

            x = self._create_epochs()
            self._classify(x)

            # reset the counters of the StreamWatchers and the FilterBank
            for sw in self.input_sws:
                sw.n_new = 0
            self.filterbank.n_new = 0

    def _create_epochs_markers(self) -> NDArray | int:
        # logger.debug(f"Loading latest markers")
        self.input_mrk_sw.update()
        if self.input_mrk_sw.n_new == 0:
            # logger.debug("Skipping epoching because no new markers")
            return 0
        markers = self.input_mrk_sw.unfold_buffer()[
            -self.input_mrk_sw.n_new :, 0
        ]  # assumes only one channel
        # starts or the epochs in seconds:
        markers_t = (
            self.input_mrk_sw.unfold_buffer_t()[-self.input_mrk_sw.n_new :]
            + self.offset_nsamples
        )
        # mask for desired markers:
        if self.markers is None:
            desired = np.ones_like(markers, dtype=bool)
        else:
            desired = np.isin(markers, self.markers)
        if desired.sum() == 0:
            # logger.debug("No new markers")
            return 0

        # logger.debug(f"Loading the latest data time stamps")
        t = self.filterbank.ring_buffer.unfold_buffer_t()[-self.filterbank.n_new :]
        starts = np.abs(markers_t[:, None] - t).argmin(axis=1)

        # compute masks
        n_times = int(self.input_window_seconds * self.input_sfreq)
        missed = starts == 0
        to_wait = starts + n_times > len(t)
        to_process = ~missed & ~to_wait
        n_missed = missed.sum()
        n_to_wait = to_wait.sum()
        n_to_process = to_process[desired].sum()
        self.input_mrk_sw.n_new = n_to_wait

        if n_missed > 0:
            logger.error(
                f"{n_missed} events could not be embedded "
                f"because t_sleep_s or processing time too long."
            )
        if missed[n_missed:].any() or to_wait[:-n_to_wait].any():
            logger.error(
                "Shuffled time stamps found in the markers stream. "
                "This case is not handled."
            )
        if n_to_process == 0:
            # logger.debug("No events can be epoched")
            return 0

        logger.debug(f"Loading the latest data samples")
        x = self.filterbank.get_data()[:, 1:33, 0]
        assert len(t) == x.shape[0]
        logger.debug(
            f"Epoching {n_to_process} events ({markers[desired & to_process]}) "
            f"of {n_times} samples each"
        )
        x = np.stack(
            [x[start : start + n_times, :].T for start in starts[desired & to_process]],
            axis=0,
            dtype=np.float32,
        )
        return x

    def _filter(self):
        if self.input_sw is None:
            logger.error(
                "Input StreamWatcher not connected, use `self.connect_input_streams` first"
            )
            return 1

        if self.input_sw.n_new == 0:
            logger.debug("No new samples to filter")
        else:
            self.filterbank.filter(
                self.input_sw.unfold_buffer()[-self.input_sw.n_new :, :],
                self.input_sw.unfold_buffer_t()[-self.input_sw.n_new :],
            )  # after this step, the buffer within filterbank has the filtered data
            self.input_sw.n_new = 0

    def _classify(self, x: NDArray):  # (batch_size, n_channel, n_times)
        if np.isnan(x).sum() > 0:
            logger.error("NaNs found after epoching")

        if self.new_sfreq is not None:
            logger.debug(f"Resampling to {self.new_sfreq} Hz")
            up = self.input_sfreq / self.new_sfreq
            x = resample(x.astype("float64"), up=up, axis=-1).astype("float32")
            if np.isnan(x).sum() > 0:
                logger.error("NaNs found after resampling")

        logger.debug(f"Computing {x.shape[0]} embedding(s)")
        y = self.classifier.predict(x)[0]
        if np.isnan(y).sum() > 0:
            logger.error("NaNs found after embedding")

        # adding 1 to distinguish prediction "0" from empty stream
        pred = [y[0] + 1]
        logger.info(f"Pushing prediction {pred[0]}")
        self.output_sw.push_sample(pred)

    def _run_loop(self, stop_event: threading.Event):

        logger.debug("Starting the run loop")
        if self.input_sw is None or self.output_sw is None or self.classifier is None:
            logger.error("OnlineDecoder not initialized, call init_all first")

        while not stop_event.is_set():
            t_start = pylsl.local_clock()
            self.update()
            t_end = pylsl.local_clock()

            # reduce sleep by processing time
            sleep_s(self.t_sleep_s - (t_end - t_start))

    def run(self) -> tuple[threading.Thread, threading.Event]:
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._run_loop,
            kwargs={"stop_event": stop_event},
        )
        thread.start()
        return thread, stop_event


def online_decoder_factory(config_path: Path = Path("./configs/decoder.toml")):
    """Factory function to create an OnlineDecoder object from a config file."""
    cfg = toml.load(config_path)

    classifier = joblib.load(cfg["online"]["classifier"]["file"])

    online_dec = OnlineDecoder(
        classifier,
        input_stream_name=cfg["online"]["input"]["lsl_stream_name"],
        output_stream_name=cfg["online"]["output"]["lsl_stream_name"],
        input_window_seconds=cfg["online"]["input"]["buffer_size_s"],
        band=cfg["online"]["classifier"].get("band", [0.5, 40]),
        eval_after_type=cfg["online"]["eval"][
            "eval_after_type"
        ],  # this si required from a config
        eval_after_s=cfg["online"]["eval"].get("eval_after_s", 1),
        eval_after_nsamples=cfg["online"]["eval"].get("eval_after_nsamples", 1),
    )

    return online_dec
