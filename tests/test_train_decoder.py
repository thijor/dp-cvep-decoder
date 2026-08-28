import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import toml

REPO = Path(__file__).resolve().parents[1]

try:
    import pyntbci
    import dareplane_utils  # noqa: F401  (imported transitively by the module)
    import pyxdf  # noqa: F401
    import cvep_decoder.train_decoder as train_decoder

    HAVE_TRAIN_DEPS = True
except ImportError:
    HAVE_TRAIN_DEPS = False


class TestTrainingResample(unittest.TestCase):
    """Dependency-free regression guard for the padding-aware resample."""

    def test_preserves_rate_with_padding(self):
        # Resampling a padded trial to (tmax - tmin + 2*pad)*fs samples and then
        # trimming pad*fs from each side must leave exactly the unpadded trial at
        # the target rate. A regressed formula (dropping the padding) yields the
        # wrong number of samples and a wrong effective rate.
        from scipy.signal import resample

        fs_in, fs_out = 512, 120
        tmin, tmax, pad_s = 0.0, 4.2, 0.5

        n_in = round((tmax - tmin + 2 * pad_s) * fs_in)
        x = np.random.default_rng(0).standard_normal((2, 3, n_in))

        num = int((pad_s + tmax - tmin + pad_s) * fs_out)  # mirrors create_classifier
        xr = resample(x, num=num, axis=2)
        pad = int(pad_s * fs_out)
        xr = xr[:, :, pad:-pad]

        self.assertEqual(xr.shape[2], int((tmax - tmin) * fs_out))  # 504 @ 120 Hz


@unittest.skipUnless(HAVE_TRAIN_DEPS, "pyntbci / dareplane_utils / pyxdf not installed")
class TestConfigInterpretation(unittest.TestCase):
    def test_classifier_meta_from_shipped_config(self):
        cfg = toml.load(REPO / "configs" / "decoder.toml")

        cmeta = train_decoder.classifier_meta_from_cfg(cfg)

        self.assertEqual(cmeta.sfreq, 120)
        self.assertEqual(cmeta.presentation_rate, 60)
        self.assertEqual(cmeta.tmin, 0.0)
        self.assertEqual(cmeta.tmax, 4.2)
        self.assertEqual(cmeta.event, "contrast")
        self.assertTrue(cmeta.onset_event)
        self.assertEqual(cmeta.stopping, "beta")
        self.assertEqual(tuple(cmeta.fband), (6, 40))

    def test_classifier_meta_roundtrips_to_dict(self):
        cmeta = train_decoder.ClassifierMeta()

        d = asdict(cmeta)  # this is what create_classifier serialises to JSON

        self.assertEqual(d["sfreq"], cmeta.sfreq)
        self.assertEqual(d["stopping"], cmeta.stopping)

    def test_get_training_data_files_globs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sub-P001_ses-S001_run-1.xdf").write_bytes(b"")
            (root / "sub-P001_ses-S001_run-2.xdf").write_bytes(b"")
            (root / "sub-P002_ses-S001_run-1.xdf").write_bytes(b"")
            (root / "notes.txt").write_text("ignore me")

            cfg = {
                "data": {
                    "data_root": str(root),
                    "training_files_glob": "sub-P001_ses-S001*.xdf",
                }
            }

            files = train_decoder.get_training_data_files(cfg)

            self.assertEqual(len(files), 2)
            self.assertTrue(all(f.suffix == ".xdf" for f in files))


@unittest.skipUnless(HAVE_TRAIN_DEPS, "pyntbci / dareplane_utils / pyxdf not installed")
class TestModelConstruction(unittest.TestCase):
    def setUp(self):
        self.cmeta = train_decoder.ClassifierMeta(
            sfreq=120,
            presentation_rate=60,
            event="contrast",
            onset_event=True,
            encoding_length=0.3,
            ctmin=0.0,
            segment_time_s=0.1,
            min_time=0.1,
            max_time=1.0,
            cr=1.0,
        )
        # A small binary stimulus of shape (n_codes, n_samples); model
        # construction does not fit, so the exact values are irrelevant.
        rng = np.random.default_rng(0)
        self.V = (rng.random((4, 2 * 120)) > 0.5).astype(float)

    def test_get_rcca_model(self):
        model = train_decoder.get_rcca_model(self.cmeta, self.V)

        self.assertIsInstance(model, pyntbci.classifiers.rCCA)
        self.assertEqual(model.fs, int(self.cmeta.sfreq))
        self.assertEqual(model.stimulus.shape, self.V.shape)

    def test_get_rcca_model_early_stop(self):
        cases = [
            ("margin", pyntbci.stopping.MarginStopping),
            ("beta", pyntbci.stopping.DistributionStopping),
            ("norm", pyntbci.stopping.DistributionStopping),
            ("accuracy", pyntbci.stopping.CriterionStopping),
            ("bds0", pyntbci.stopping.BayesStopping),
        ]
        for stopping, cls in cases:
            with self.subTest(stopping=stopping):
                self.cmeta.stopping = stopping
                model = train_decoder.get_rcca_model_early_stop(self.cmeta, self.V)
                self.assertIsInstance(model, cls)
                self.assertIsInstance(model.estimator, pyntbci.classifiers.rCCA)

    def test_get_rcca_model_early_stop_unknown_raises(self):
        self.cmeta.stopping = "does-not-exist"
        with self.assertRaises(ValueError):
            train_decoder.get_rcca_model_early_stop(self.cmeta, self.V)


if __name__ == "__main__":
    unittest.main()
