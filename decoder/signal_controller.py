import threading

import numpy as np
from numpy.typing import NDArray
import pylsl
from mne.filter import resample

from dareplane_utils.general.time import sleep_s
from dareplane_utils.signal_processing.filtering import FilterBank
from dareplane_utils.stream_watcher.lsl_stream_watcher import StreamWatcher
from pyntbci.classifiers import rCCA

from decoder.utils.logging import logger

logger.setLevel(10)

class SignalEmbedder:
    """Class to embed windows of signal using a sklearn transformer.

    Parameters
    ----------

    classifier: rcca
        classifier object to use for decoding
    input_stream_name: str
        Name of the input stream (containing the signal).
    output_stream_name: str
        Name of the output stream (containing the embeddings).
    input_window_seconds: float
        Temporal size of the window that should be passed to the  model.
    new_sfreq: float | None
        If not None, resample the signal to this frequency before passing it to the model.
    band: tuple[float, float]
        Band to filter the signal before passing it to the model.
    t_sleep: float
        Time to sleep between updates.
    marker_stream_name: str | None
        Optional name of the marker stream. If not None, the projections will be synchronised
        with the markers.
    markers: list[str]
        List of markers to create projections for. If None, all markers are used.
        Only used if marker_stream_name is not None.
    offset: float
        Offset of the epochs wrt the markers. Only used if marker_stream_name is not None.
    """

    def __init__(
            self,
            classifier: rCCA,
            input_stream_name: str,
            output_stream_name: str,
            input_window_seconds: float,
            new_sfreq: float | None = None,
            band: tuple[float, float] = (0.5, 40),
            t_sleep: float = 0.1,
            marker_stream_name: str | None = None,
            markers: list[str] | None = None,
            offset: float = 0,
    ):
        self.model = classifier
        self.input_stream_name = input_stream_name
        self.output_stream_name = output_stream_name
        self.input_window_seconds = input_window_seconds
        self.new_sfreq = new_sfreq
        self.band = band
        self.t_sleep = t_sleep
        self.marker_stream_name = marker_stream_name
        self.markers = markers
        self.offset = offset

        self.data_sw = None
        self.mrk_sw = None
        self.fb = None
        self.emb_so = None
        self.signal_sfreq = None
        self.chs_info = None

    def get_self(self):
        return self

    @property
    def buffer_size_s(self):
        # we take a larger buffer for margin:
        return 3 * (max(self.input_window_seconds, self.t_sleep) + abs(self.offset))

    @property
    def model_sfreq(self):
        return self.signal_sfreq if self.new_sfreq is None else self.new_sfreq

    @property
    def n_outputs(self):
        x = np.zeros((1, len(self.chs_info), int(self.input_window_seconds * self.model_sfreq)),
                     dtype=np.float32)
        y = self.model.predict(x)
        # assert y.ndim == 1
        # assert y.shape[0] == 1
        return y.shape[1]

    def set_signal_info(self):
        self.signal_sfreq = self.data_sw.inlet.info().nominal_srate()
        self.chs_info = [
            dict(ch_name=ch_name, type="EEG")
            for ch_name in self.data_sw.channel_names
        ]

    def connect_input_streams(self):
        logger.info(f'Connecting to input stream "{self.input_stream_name}"')
        self.data_sw = StreamWatcher(
            self.input_stream_name, buffer_size_s=self.buffer_size_s, logger=logger)

        self.data_sw.connect_to_stream()

        # set signal_sfreq and chs_info
        self.set_signal_info()

        if self.marker_stream_name is None:
            return 0

        logger.info(f'Connecting to marker stream "{self.marker_stream_name}"')
        self.mrk_sw = StreamWatcher(
            self.marker_stream_name, buffer_size_s=self.buffer_size_s, logger=logger)
        self.mrk_sw.connect_to_stream()
        if len(self.mrk_sw.channel_names) != 1:
            logger.error("The marker stream should have exactly one channel")
        return 0

    def create_model(self):
        logger.info("Creating the model and the filter bank.")
        self.fb = FilterBank(
            bands={"band": self.band},
            sfreq=self.signal_sfreq,
            output="signal",
            n_in_channels=len(self.chs_info),
            filter_buffer_s=self.buffer_size_s,
        )
        return 0

    def create_output_stream(self):
        logger.info(f'Creating the output stream "{self.output_stream_name}"')
        info = pylsl.StreamInfo(
            self.output_stream_name, "MISC", channel_count=1,
            nominal_srate=pylsl.IRREGULAR_RATE, channel_format="int32",
            source_id="signal_embedding",  # TODO should it be more unique?
        )
        self.emb_so = pylsl.StreamOutlet(info)
        return 0

    def init_all(self):
        self.connect_input_streams()
        self.create_model()
        self.create_output_stream()
        return 0

    def update(self):
        # logger.debug(f"Start update")
        if self._filter():
            return 1
        if self.marker_stream_name is None:
            x = self._create_epoch_latest()
        else:
            x = self._create_epochs_markers()
        if isinstance(x, int):
            return x
        return self._project(x)

    def _create_epoch_latest(self) -> NDArray | int:
        logger.debug(f"Loading the latest data samples")
        # FilterBank returns (n_times, n_channel, n_bands)
        x = self.fb.get_data()
        n_times = int(self.input_window_seconds * self.signal_sfreq)
        if x.shape[0] < n_times:
            logger.debug(
                f"Skipping embedding because not enough samples in buffer ({x.shape[0]}/{n_times})")
            return 0

        logger.debug(f"Creating one epoch from the {n_times} latest samples")
        x = x[-n_times:, :, 0]
        # Transpose and add batch dim
        x = x.T[None, :, :]  # (1, n_channel, n_times)

        return x

    def _create_epochs_markers(self) -> NDArray | int:
        # logger.debug(f"Loading latest markers")
        self.mrk_sw.update()
        if self.mrk_sw.n_new == 0:
            # logger.debug("Skipping epoching because no new markers")
            return 0
        markers = self.mrk_sw.unfold_buffer()[-self.mrk_sw.n_new:, 0]  # assumes only one channel
        # starts or the epochs in seconds:
        markers_t = self.mrk_sw.unfold_buffer_t()[-self.mrk_sw.n_new:] + self.offset
        # mask for desired markers:
        if self.markers is None:
            desired = np.ones_like(markers, dtype=bool)
        else:
            desired = np.isin(markers, self.markers)
        if desired.sum() == 0:
            # logger.debug("No new markers")
            return 0

        # logger.debug(f"Loading the latest data time stamps")
        t = self.fb.ring_buffer.unfold_buffer_t()[-self.fb.n_new:]
        starts = np.abs(markers_t[:, None] - t).argmin(axis=1)

        # compute masks
        n_times = int(self.input_window_seconds * self.signal_sfreq)
        missed = starts == 0
        to_wait = starts + n_times > len(t)
        to_process = ~missed & ~to_wait
        n_missed = missed.sum()
        n_to_wait = to_wait.sum()
        n_to_process = to_process[desired].sum()
        self.mrk_sw.n_new = n_to_wait

        if n_missed > 0:
            logger.error(f"{n_missed} events could not be embedded "
                         f"because t_sleep or processing time too long.")
        if missed[n_missed:].any() or to_wait[:-n_to_wait].any():
            logger.error("Shuffled time stamps found in the markers stream. "
                         "This case is not handled.")
        if n_to_process == 0:
            # logger.debug("No events can be epoched")
            return 0

        logger.debug(f"Loading the latest data samples")
        x = self.fb.get_data()[:, 1:33, 0]
        assert len(t) == x.shape[0]
        logger.debug(
            f"Epoching {n_to_process} events ({markers[desired & to_process]}) "
            f"of {n_times} samples each")
        x = np.stack([
            x[start: start + n_times, :].T
            for start in starts[desired & to_process]
        ], axis=0, dtype=np.float32)
        return x

    def _filter(self):
        if self.data_sw is None:
            logger.error("SignalEmbedder not initialized, call init_all first")
            return 1
        # Grab latest samples
        self.data_sw.update()
        if self.data_sw.n_new == 0:
            logger.debug("Skipping filtering because no new samples")
            return 0

        # logger.debug(f"Filtering {self.data_sw.n_new} new samples")
        self.fb.filter(
            # look back only new data
            self.data_sw.unfold_buffer()[-self.data_sw.n_new:, :],
            # and this is getting the times
            self.data_sw.unfold_buffer_t()[-self.data_sw.n_new:],
        )  # after this step, the buffer within fb has the filtered data
        self.data_sw.n_new = 0
        return 0

    def _project(self, x: NDArray):  # (batch_size, n_channel, n_times)
        if self.emb_so is None or self.model is None:
            logger.error("SignalEmbedder not initialized, call init_all first")
            return 1
        if np.isnan(x).sum() > 0:
            logger.error("NaNs found after epoching")

        if self.new_sfreq is not None:
            logger.debug(f"Resampling to {self.new_sfreq} Hz")
            up = self.signal_sfreq / self.new_sfreq
            x = resample(x.astype("float64"), up=up, axis=-1).astype("float32")
            if np.isnan(x).sum() > 0:
                logger.error("NaNs found after resampling")

        logger.debug(f"Computing {x.shape[0]} embedding(s)")
        y = self.model.predict(x)[0]
        if np.isnan(y).sum() > 0:
            logger.error("NaNs found after embedding")

        #adding 1 to distinguish prediction "0" from empty stream
        pred = [y[0]+1]
        logger.info(f"Pushing prediction {pred[0]}")
        self.emb_so.push_sample(pred)
        return 0

    def _run_loop(self, stop_event: threading.Event):
        logger.debug("Starting the run loop")
        if self.data_sw is None or self.emb_so is None or self.model is None:
            logger.error("SignalEmbedder not initialized, call init_all first")
            return 1
        while not stop_event.is_set():
            t_start = pylsl.local_clock()
            self.update()
            t_end = pylsl.local_clock()

            # reduce sleep by processing time
            sleep_s(self.t_sleep - (t_end - t_start))
        return 0

    def run(self) -> tuple[threading.Thread, threading.Event]:
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._run_loop,
            kwargs={"stop_event": stop_event},
        )
        thread.start()
        return thread, stop_event