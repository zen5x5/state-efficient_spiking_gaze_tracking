# EBV-Eye ANN/SNN training and testing

This tree contains models, training, evaluation, cache loading, tests and final annotations. It uses the frozen, training-ready eye0 caches for users 4–27. Final annotations are included in `annotations/`. Raw-data preprocessing and AI boundary selection are outside this release.

Clone this standalone repository before running the commands below:

```bash
git clone https://github.com/zen5x5/state-efficient_spiking_gaze_tracking.git
cd state-efficient_spiking_gaze_tracking
```

## Environment

Python 3.11, CUDA-capable PyTorch, Sinabs and Exodus. The tested direct package versions are in `requirements.txt`. The original training environment used PyTorch 2.13.0+cu130 on NVIDIA L40S, Sinabs 3.1.3 and Exodus 2.0.0. Train SNNs with Exodus on a GPU; `--backend sinabs --device cpu` is available for checkpoint evaluation.

```bash
python -m pip install -r requirements.txt
export EBV_EYE_DATA_ROOT="$HOME/ebv-eye-local"
export EBV_EYE_CACHE_ROOT="$EBV_EYE_DATA_ROOT/cache/formal_eye0"
export EBV_EYE_RESULT_ROOT="$HOME/ebv-eye-results"
```

## Download training data

`datasets/formal_eye0.json` contains file and archive SHA-256 values for 48 task/subject records. Caches have fixed train/validation/test membership, 400-event count frames of shape 2×96×128 and physical train overlap samples. The downloader validates every file before installation.

The 48 archives (321.8 MiB total) will be available in the [formal eye0 cache release](https://github.com/zen5x5/state-efficient_spiking_gaze_tracking/releases/tag/formal-eye0-training-cache) once uploaded. The download catalog records archive and file SHA-256 values.

```bash
python -m code.download_dataset --users 15 --tasks pursuit saccade --output "$EBV_EYE_CACHE_ROOT"
# All 24 users: --users $(seq 4 27)
# Local archive mirror: add --archive-root /path/to/formal-training-cache-assets
# Other published asset mirror: add --base-url https://your-host/cache-assets
```

## Train

The default route uses ordinary loss, all-physical-source 200+200-event recombination, batch size 16, Adam learning rate 0.001, validation-based learning-rate scheduling and checkpoint selection. ANN training lasts 200 epochs. Final SNN training uses T=2 repeated full frames, alpha=0.75, 100 full-state epochs, then continuation to epoch 200 after setting L1/L2 stateless. Seeds 2021/2022/2023 are supported.

```bash
python -m code.retrain --task pursuit --model snn --user 15 --seeds 2021 \
  --output-root "$EBV_EYE_RESULT_ROOT" --device cuda:0
python -m code.retrain --task saccade --model snn --user 15 --seeds 2021 \
  --output-root "$EBV_EYE_RESULT_ROOT" --device cuda:0
# ANN: replace --model snn with --model ann.
# Inspect the exact commands without training: add --dry-run.
```

## Test the selected checkpoint

Training writes `configuration.json`, `best_checkpoint.pt`, `training_history.json` and `summary.json`. It reports the complete held-out test split after checkpoint selection using validation only. A standalone evaluation can be repeated as follows:

```bash
python -m code.test_checkpoint --task saccade --model snn \
  --result-dir "$EBV_EYE_RESULT_ROOT/saccade/snn/user15/eye0/seed2021/l1_l2_stateless_epoch200" \
  --cache-dir "$EBV_EYE_CACHE_ROOT/saccade/fixed_event_window_400_stride200_96x128/user15/eye0" \
  --device cuda:0 --backend exodus
```

`benchmarks/reference_results.json` gives every subject's original seed2021 validation/test reference and provenance hashes. The 24-subject test means are 0.823945° for pursuit and 1.325509° for saccade, continuous regression. These are per-subject models; a user15 run should be compared with user15's reference, not the 24-subject mean. Hardware/software and training seed can change results; the release fixes the inputs, model and training contract.

## Tests

After installing the user15 caches:

```bash
python -m code.tests_training_release
python -m code.pursuit.fixed_event_window.ann.tests
python -m code.pursuit.fixed_event_window.snn.tests
python -m code.saccade.fixed_event_window.ann.tests
python -m code.saccade.fixed_event_window.snn.tests
```

Some modules retain tested experimental options. The reproduction commands above select the formal continuous-output, ordinary-loss route.

## Publish the prepared cache assets

The upload command requires an existing GitHub token with repository Contents write access (see [GitHub release API documentation](https://docs.github.com/en/rest/releases/releases)). The token is read from GH_TOKEN or GITHUB_TOKEN and is never written to the catalog. Push the clean release branch first, then upload the 48 prepared archives from the archive directory supplied with the release handoff:

```bash
git push origin main
python -m code.publish_training_cache \
  --catalog datasets/formal_eye0.json \
  --archive-root /path/to/formal-training-cache-assets \
  --output-catalog datasets/formal_eye0.json
```

The publisher creates a draft release, verifies all asset digests, and publishes it only after all uploads finish. Commit and push the catalog populated with the actual download URLs. An alternate repository can be selected with --repository; no repository visibility is changed.

The release was checked in an isolated tree: all four model test suites and four release checks passed; cached user15 archives were loaded by the downloader; ANN and saccade SNN one-step training/checkpoint smoke checks passed. The standalone CPU test returned 0.746519° pursuit and 1.252111° saccade on the original user15 checkpoints, differing from the original GPU reference by 0.001602° and 0.002199°. These are checkpoint replays, not newly completed 200-epoch training runs.

The publisher refreshes final download URLs after the draft is published.
