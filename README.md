# Drone Audio Detection
### Building and Stress-Testing an Audio-Based Classifier

> **Status: in progress.** This README reflects the current state of the project and is being updated as work continues. Sections marked `[IN PROGRESS]` are not yet complete.

---

## Abstract

This project trains a Random Forest classifier to detect drones from audio using MFCC-based features, then goes beyond a single accuracy number to understand *when and why* the model generalizes. A baseline model reaches strong benchmark metrics (ROC-AUC 0.92, PR-AUC 0.84), but live testing against a real drone and a directional microphone revealed a clear **sim-to-real gap**: the model fails to reliably detect real-world drone audio despite strong offline performance. The rest of this project investigates that gap systematically — through cross-drone generalization analysis, error analysis, and a quantified real-world test set — rather than treating it as a footnote. `[IN PROGRESS]`

---

## Project Goal

Build an audio-based drone detection classifier and rigorously evaluate its real-world reliability — not just its offline benchmark performance. The focus is on understanding generalization limits, dataset quality issues, and the gap between benchmark and deployment conditions, which are central concerns for any real-world detection system.

---

## Pipeline Architecture

```
Raw audio (HuggingFace dataset)
        │
        ▼
  Preprocessing  (preprocess.py)
  - padding with ambient noise
  - pre-cut augmentation (background mix, noise injection, reverb)
  - sliding-window segmentation (1.0s window, 0.25s stride)
  - post-cut augmentation (gain, time shift, band-pass, time stretch, pitch shift)
        │
        ▼
  Feature Extraction  (features.py)
  - MFCC (13 coefficients) + delta + delta-delta
  - mean + std aggregation per coefficient
  - 78-dimensional feature vector per segment
        │
        ▼
  Model Training  (rf.py)
  - Random Forest, class-weighted
  - Group-aware train/val/test split (GroupShuffleSplit)
        │
        ▼
  Evaluation  (evaluate.py)
  - Accuracy, Precision/Recall/F1, ROC-AUC, PR-AUC
```

All stages are orchestrated through a single Hydra-configured entry point (`run.py`), with parameters defined in `configs/`.

---

## Key Design Decisions

Every non-obvious choice in this pipeline was made deliberately. A few worth highlighting:

- **MFCC over mel spectrogram for feature extraction.** Mel spectrograms preserve more raw information and pair naturally with CNNs, but a flat MFCC feature vector is a better fit for a tree-based classifier like Random Forest, and requires no additional flattening step that would discard the spectrogram's 2D structure.
- **Delta and delta-delta coefficients.** Drone rotor noise is quasi-periodic; delta features capture how MFCCs change frame-to-frame, which is informative for that kind of signal. This triples the feature vector from 26 to 78 dimensions.
- **Ambient-noise padding instead of zero-padding.** Many source clips are shorter than the 1-second analysis window. Padding with silence would teach the model to associate "silence" with "end of clip"; padding with real ambient noise avoids that artifact.
- **Group-aware splitting.** Segments generated via the sliding window from the same source file are highly correlated. A naive random split would leak near-duplicate segments across train/test, inflating reported performance. `GroupShuffleSplit` keeps all segments from a given source file within a single split.
- **Class-weighted training, not resampling.** The dataset is heavily imbalanced (see below). SMOTE-style oversampling was considered and rejected — interpolating between audio feature vectors doesn't correspond to a physically meaningful audio signal. `class_weight='balanced'` was used instead.
- **PR-AUC over ROC-AUC as the primary metric.** With substantial class imbalance, ROC-AUC can look deceptively strong. PR-AUC is a more honest signal of performance on the minority (in this case, differently-distributed) class.

---

## Dataset

[`geronimobasso/drone-audio-detection-samples`](https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples) (HuggingFace) — approximately 180,000 samples at 16kHz, aggregated from multiple drone and ambient-noise sources.

- Roughly an even split between drone and no-drone samples, but with a notable asymmetry: **no-drone clips are, on average, considerably longer** than drone clips. This shaped several preprocessing decisions (ambient-noise padding, sliding-window segmentation).
- No source-level metadata is available — individual sub-datasets are aggregated without a source label per sample. This is a known limitation and is addressed later via unsupervised clustering (see [Cross-Drone Generalization](#cross-drone-generalization-in-progress)).
- Label quality is imperfect: some drone clips are recorded from distant microphones and are only faintly audible, raising the possibility of borderline or noisy labels. See [Known Limitations](#known-limitations).

---

## Baseline Results (Random Forest)

Trained on the full dataset (~180k samples → segmented feature vectors), class-weighted, group-aware split.

| Metric | Test | Validation |
|---|---|---|
| Accuracy | 0.86 | 0.85 |
| ROC-AUC | 0.92 | 0.92 |
| PR-AUC | 0.84 | 0.84 |

*(Detailed classification report and confusion matrix available in `evaluate.py` output / MLflow logs.)*

**What this doesn't tell you:** these numbers describe performance on held-out segments from the *same* dataset. They say nothing about how the model behaves on audio it wasn't trained on — which turned out to matter a great deal.

---

## The Sim-to-Real Gap

Real-time inference was tested against:
- A live DJI Mini 3 drone, recorded via laptop microphone and a directional microphone
- Several drone audio clips sourced from YouTube

In both cases, the model **failed to reliably detect the drone**, while frequently misclassifying quiet ambient noise as "drone." This is a stark contrast to the strong benchmark metrics above, and is the central finding this project investigates.

### Quantified Real-World Test `[IN PROGRESS]`

A small labeled test set recorded with a directional microphone (drone-in-flight and ambient/no-drone clips) is being used to replace this anecdotal observation with hard numbers, run through the same `load_audio → extract_features → model.predict` pipeline used in `inference.py`.

---

## Cross-Drone Generalization `[IN PROGRESS]`

The dataset provides no source labels, so a true leave-one-source-out evaluation isn't possible directly. As an approximation:

1. K-Means clustering on MFCC feature vectors to identify pseudo-source clusters
2. Manual/qualitative verification that clusters correspond to meaningfully different audio characteristics (not just an assumption — clusters are spot-checked by listening to samples from each)
3. Leave-one-cluster-out training and evaluation, repeated across clusters, to measure how performance degrades on an unseen "drone type" — reported per-cluster, not as a single averaged number, to show variance rather than hide it

This is intended to connect directly to the sim-to-real gap observed above: does the model generalize to drone characteristics it hasn't seen, or has it primarily learned the specific drones present in this dataset?

---

## Error Analysis `[IN PROGRESS]`

- High-confidence misclassifications (cases where the model was highly certain and wrong) are inspected individually, with spectrograms, to understand *what* is being confused for what (e.g. engine/motor noise misclassified as drone).
- Feature importance from the trained Random Forest, connected back to MFCC coefficients and blade-pass-frequency findings from EDA.
- Learning curve (performance vs. training set size), to check whether the model is data-limited.

---

## Known Limitations

- **Label quality:** some drone clips in the source dataset are recorded from distant microphones with faint, borderline-audible drone sound. This introduces potential label noise that isn't correctable without re-listening to the full dataset.
- **No source metadata:** the dataset aggregates multiple original sources without labeling which sample came from which. Cross-drone generalization analysis is therefore approximated via clustering, not verified ground truth.
- **Random Forest with a fixed decision threshold:** the default 0.5 threshold is not tuned for this use case, where missing a drone is arguably worse than a false alarm. Threshold tuning is planned but not yet reflected in the reported metrics above.
- **No fine-tuning on target hardware:** the model was trained entirely on the source dataset's recording conditions; no fine-tuning on the actual microphone hardware used for live testing was performed, which likely contributes to the sim-to-real gap.

---

## What I'd Do Differently

- Use a dataset with proper per-sample source labels from the start, enabling a true leave-one-source-out split rather than a clustering approximation.
- Collect a larger, purpose-built real-world validation set before trusting benchmark metrics at all.
- Fine-tune (or at least validate) on the actual target microphone hardware early in the project, rather than discovering the domain gap late.
- Give a properly-tuned CNN on mel spectrograms a fair comparison, rather than relying solely on a classical ML baseline.

---

## Reproducibility

- Environment and dependencies are managed with [`uv`](https://github.com/astral-sh/uv) (`pyproject.toml` / `uv.lock`).
- Configuration is managed with [Hydra](https://hydra.cc/) (`configs/`), with experiment tracking via [MLflow](https://mlflow.org/). `[IN PROGRESS]`
- All randomness is seeded (`config.yaml: seed`).

### Setup

```bash
uv sync
```

### Running the pipeline

```bash
uv run src/run.py
```

Parameters (feature extraction, model hyperparameters, preprocessing) can be overridden via Hydra, e.g.:

```bash
uv run src/run.py model.n_estimators=200
```

---

## License

This project's code is licensed under the MIT License — see [`LICENSE`](LICENSE).

The training dataset (`geronimobasso/drone-audio-detection-samples`) is subject to its own license on HuggingFace; refer to the dataset page for terms.