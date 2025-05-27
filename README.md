# Dareplane c-VEP Decoder

This is a module for the [Dareplane](https://github.com/bsdlab/Dareplane) project. It provides a decoding module to classify the code-modulated visual evoked potential (c-VEP) from the EEG.

## Installation

To download the dp-cvep-speller, use:

    git clone https://github.com/thijor/dp-cvep-decoder.git

Make sure that requirements are installed, ideally in a separate conda environment:

    conda create --name dp-cvep-decoder python=3.10
    conda activate dp-cvep-decoder
    pip install -r requirements.txt

## Getting started

To run the dp-cvep-speller module in isolation, use:

    python -m cvep_decoder.decoder.py

This will run a minimal example using defaults as specified in `configs/decoder.toml`.

## Notes

Be aware of the interplay if the following two timing parameters:

```toml
[online]
sleep_s = 0.1                            # time between updates in seconds

[online.eval]
eval_after_type = 'time'
eval_after_s = 0.1                        # time after which the model is evaluated
```

If both are set as in the example above, you might get an unexpected behavior. This is due to the fact that system sleeps are never precise. Which could result in an update cycle producing a prediction and then sleeping for e.g. `90ms` to match the target `100ms` update cycles defined by `sleep_s=0.1`. After the sleep, the next update cycle will start. It is possible however, that at this very time, the LSL stream watcher will have collected only samples for `99ms` worth of data (the `eval_after_s` is internally interpreted as a required sample number by: `int(eval_after_s * sampling_freq)` for the input LSL stream. With these insufficient number of samples, another sleep for about `100ms` is triggered. Subsequently, data is processed for about `199ms` in the next update cycle, which will then evaluate the classifier. The simplest solution is to have `sleep_s`, i.e., the update cycle time according to your target, and having `eval_after_s` smaller than `sleep_s`. The decoder will then produce decodings in relatively stable intervals of `sleep_s`. The `eval_after_s` parameter would be useful, if evaluation should happen much slower than the update cycle (`sleep_s`).

## TODO

- [ ] The git file for this repo contains some very large blobs since the first commit. We should remove them, which however will overwrite a few commit hashes. Everyone how has pulled the repo would need to re-pull. A potential solution would then be to use the following:

```
pip install git-filter-repo

# Identify the large objects:

git verify-pack -v .git/objects/pack/pack-*.idx | sort -k 3 -n | tail -5

# Remove the large files from history:

git filter-repo --strip-blobs-bigger-than 10M  # Example: remove blobs >10MB
```
