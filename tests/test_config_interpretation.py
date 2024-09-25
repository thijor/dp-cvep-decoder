# Test to check that the config gets read correctly

import joblib
import numpy as np
import toml

from cvep_decoder.online_decoding import online_decoder_factory
from tests.shared import provide_config_toml, provide_joblib_model


def test_creation_with_default_config(provide_config_toml, provide_joblib_model):
    file = provide_joblib_model  # this ensures that a model file is generated
    cfg_file = provide_config_toml

    print(cfg_file)

    online_decoder = online_decoder_factory(cfg_file)
    cfg = toml.load(cfg_file)

    classifier = joblib.load(cfg["online"]["classifier"]["file"])

    assert online_decoder is not None
    assert online_decoder.buffer_size_s == 3
    assert online_decoder.classifier.fs == classifier.fs
    assert np.allclose(online_decoder.classifier.stimulus, classifier.stimulus)
    assert (
        online_decoder.selected_channels == cfg["online"]["input"]["selected_channels"]
    )
