import types
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

try:
    import pylsl  # noqa: F401
    import fire  # noqa: F401
    import dareplane_utils  # noqa: F401
    import cvep_decoder.online_decoding as online_decoding

    HAVE_ONLINE_DEPS = True
except ImportError:
    HAVE_ONLINE_DEPS = False


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #
class FakeMarkerSW:
    """Minimal stand-in for a marker StreamWatcher."""

    def __init__(self, markers, times):
        self._buf = np.array(markers, dtype=object).reshape(-1, 1)
        self._t = np.asarray(times, dtype=float)
        self.n_new = len(markers)

    def unfold_buffer(self):
        return self._buf

    def unfold_buffer_t(self):
        return self._t


class FakeFilterBank:
    """Minimal stand-in for a FilterBank; get_data returns (n, n_ch, n_bands)."""

    def __init__(self, data):
        self._data = data

    def get_data(self):
        return self._data


def make_decoder():
    """A bare OnlineDecoder (no streams connected, no model loaded)."""
    return online_decoding.OnlineDecoder(
        decoder_file="model.joblib",
        decoder_meta_file="model_meta.json",
        marker_stream_name="markers",
        data_stream_name="eeg",
        decoder_stream_name="decoder",
        buffer_size_s=5.0,
        padding_size_s=0.0,
        start_eval_marker="start_trial",
        t_sleep_s=0.1,
        max_eval_time_s=10,
        selected_channels=None,
    )


@unittest.skipUnless(HAVE_ONLINE_DEPS, "pylsl / fire / dareplane_utils not installed")
class TestOnlineDecoder(unittest.TestCase):
    def setUp(self):
        self.dec = make_decoder()

    # ---- factory / config ------------------------------------------------- #
    def test_factory_reads_shipped_config(self):
        dec = online_decoding.online_decoder_factory(
            REPO / "configs" / "decoder.toml", preload=False
        )

        self.assertEqual(dec.data_stream_name, "BioSemi")
        self.assertEqual(dec.marker_stream_name, "cvep-speller-stream")
        self.assertEqual(dec.decoder_stream_name, "cvep-decoder-stream")
        self.assertEqual(dec.padding_size_s, 0.5)
        self.assertEqual(dec.t_sleep_s, 0.1)
        self.assertEqual(dec.start_eval_marker, "start_trial")
        self.assertEqual(
            dec.selected_channels, ["EX1", "EX2", "EX3", "EX4", "EX5", "EX6"]
        )

    # ---- _resample -------------------------------------------------------- #
    def test_resample_downsamples_to_classifier_rate(self):
        self.dec.input_sfreq = 500
        self.dec.classifier_input_sfreq = 250
        self.dec.padding_size_s = 0.0

        x = np.random.default_rng(0).standard_normal((1, 3, 1000))
        xs = self.dec._resample(x)

        self.assertEqual(xs.shape[:2], (1, 3))
        self.assertEqual(xs.shape[2], 500)  # 1000 * 250 / 500

    def test_resample_removes_leading_padding(self):
        self.dec.input_sfreq = 500
        self.dec.classifier_input_sfreq = 250
        self.dec.padding_size_s = 0.5  # -> pad = int(250 * 0.5) = 125 samples

        x = np.random.default_rng(0).standard_normal((1, 3, 1000))
        xs = self.dec._resample(x)

        self.assertEqual(xs.shape[2], 500 - 125)

    # ---- _classify -------------------------------------------------------- #
    def _wire_classify(self, prediction):
        """Attach a fake classifier + outlet; return the list receiving pushes."""
        pushed = []
        self.dec.classifier_input_sfreq = 250
        self.dec.t_sleep_s = 0.1  # min samples to classify = int(0.1 * 250) = 25
        self.dec.is_decoding = True
        self.dec.output_sw = types.SimpleNamespace(
            push_sample=lambda s: pushed.append(s)
        )
        self.dec.classifier = types.SimpleNamespace(
            predict=lambda x: np.array([prediction])
        )
        return pushed

    def test_classify_pushes_confident_prediction(self):
        pushed = self._wire_classify(prediction=3)

        self.dec._classify(np.zeros((1, 3, 300)))  # 300 >= 25

        self.assertEqual(len(pushed), 1)
        self.assertEqual(int(pushed[0][0]), 3)
        self.assertFalse(self.dec.is_decoding)

    def test_classify_ignores_negative_prediction(self):
        pushed = self._wire_classify(prediction=-1)

        self.dec._classify(np.zeros((1, 3, 300)))

        self.assertEqual(pushed, [])
        self.assertTrue(self.dec.is_decoding)

    def test_classify_skips_when_insufficient_data(self):
        pushed = []
        self.dec.classifier_input_sfreq = 250
        self.dec.t_sleep_s = 0.1
        self.dec.is_decoding = True
        self.dec.output_sw = types.SimpleNamespace(
            push_sample=lambda s: pushed.append(s)
        )
        calls = {"n": 0}

        def predict(x):
            calls["n"] += 1
            return np.array([3])

        self.dec.classifier = types.SimpleNamespace(predict=predict)

        self.dec._classify(np.zeros((1, 3, 10)))  # 10 < 25 -> skipped

        self.assertEqual(pushed, [])
        self.assertEqual(calls["n"], 0)  # classifier not even called
        self.assertTrue(self.dec.is_decoding)

    # ---- check_if_decoding_should_start ----------------------------------- #
    def test_start_decoding_sets_scalar_onset_time(self):
        self.dec.start_eval_marker = "start_trial"
        self.dec.input_mrk_sw = FakeMarkerSW(
            markers=["foo", "start_trial", "bar"], times=[10.0, 11.0, 12.0]
        )

        self.dec.check_if_decoding_should_start()

        self.assertTrue(self.dec.is_decoding)
        self.assertEqual(self.dec.input_mrk_sw.n_new, 0)
        # regression: onset time must be a scalar, not an array
        self.assertEqual(np.ndim(self.dec.start_eval_time), 0)
        self.assertEqual(float(self.dec.start_eval_time), 11.0)

    def test_no_start_without_marker(self):
        self.dec.start_eval_marker = "start_trial"
        self.dec.input_mrk_sw = FakeMarkerSW(
            markers=["foo", "bar"], times=[10.0, 11.0]
        )

        self.dec.check_if_decoding_should_start()

        self.assertFalse(self.dec.is_decoding)

    # ---- _create_epoch ---------------------------------------------------- #
    def _wire_epoch(self, n_samples, n_ch, sfreq):
        data = np.random.default_rng(0).standard_normal((n_samples, n_ch, 1))
        self.dec.filterbank = FakeFilterBank(data)
        t = np.arange(n_samples) / sfreq
        self.dec.input_sw = types.SimpleNamespace(unfold_buffer_t=lambda: t)
        self.dec.input_sfreq = sfreq
        return t

    def test_epoch_clamps_negative_start_index(self):
        n, ch, fs = 100, 3, 100
        t = self._wire_epoch(n, ch, fs)
        self.dec.padding_size_s = 0.5  # pad = 50 samples
        self.dec.start_eval_time = t[10]  # idx = 10, idx - pad = -40 -> clamp to 0

        x = self.dec._create_epoch()

        self.assertEqual(x.shape, (1, ch, n))  # whole buffer, no wrap-around

    def test_epoch_applies_padding_offset(self):
        n, ch, fs = 100, 3, 100
        t = self._wire_epoch(n, ch, fs)
        self.dec.padding_size_s = 0.1  # pad = 10 samples
        self.dec.start_eval_time = t[60]  # idx = 60, idx - pad = 50

        x = self.dec._create_epoch()

        self.assertEqual(x.shape, (1, ch, n - 50))  # from sample 50 to the end


if __name__ == "__main__":
    unittest.main()
