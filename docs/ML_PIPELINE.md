# Document ML Pipeline

This project extends the existing forensic modules in stages. The datasets and trained weights
are intentionally not included in the repository because they may contain identity or biometric
data and have separate usage terms.

## Current implementation

- `src/datasets/manifest.py` validates images, records dimensions and mode, hashes files, and
  writes deterministic CSV manifests.
- `src/datasets/midv2020.py`, `idnet.py`, and `fgnet.py` add dataset-specific labels and grouping
  conventions.
- `src/analysis/mrz.py` parses TD1, TD2, and TD3 ICAO MRZ layouts and validates check digits.
- `src/models/tamper_detector.py` exposes a stable tamper prediction contract and an offline ELA
  plus copy-move baseline.
- `src/models/face_verification.py` provides consent-gated embedding comparison only. It does not
  extract faces or claim identity proof without a separately reviewed model.
- `src/models/fusion.py` combines available evidence with configurable weights and renormalizes
  missing optional signals.
- `src/inference/pipeline.py` produces a stable JSON-compatible report contract.

## Data preparation

Keep downloaded data outside Git:

```text
data/
├── raw/
├── processed/
├── cache/
└── manifests/
```

Build manifests after downloading each dataset through its approved source. Use group IDs to
prevent leakage: a MIDV video/document, an IDNet source document, or an FG-NET subject must not
appear in both training and validation/test splits.

The local preparation command is:

```powershell
uv run python -m src.training.prepare_manifests idnet data/raw/idnet data/manifests/idnet.csv --split train
uv run python -m src.training.prepare_manifests idnet data/raw/idnet data/manifests/idnet-validation.csv --split validation
```

After both splits are represented in one manifest, run the offline EfficientNet baseline:

```powershell
uv run python -m src.training.train_tamper data/manifests/idnet.csv --epochs 1 --output weights/idnet-baseline.pt
```

The trainer uses random initialization by default and never downloads weights implicitly. Increase
epochs only after checking the dataset balance, validation split, and available GPU memory.

The adapters are deliberately conservative. If a dataset's directory naming differs from the
assumptions, inspect and correct the manifest labels before training rather than silently guessing.

## Training sequence

1. Establish OCR and MRZ parsing baselines on MIDV-2020.
2. Train an EfficientNet-B0 or ConvNeXt-Tiny tamper classifier on IDNet.
3. Add segmentation only when manipulation masks are available and verified.
4. Train or adopt a reviewed ArcFace-style encoder for face embeddings; use FG-NET for age-aware
   robustness experiments, not as a document-authenticity label.
5. Compare every learned model with the existing ELA and ORB/RANSAC baselines.
6. Register model and dataset versions in the inference report before enabling them in the UI.

Track precision, recall, F1, ROC-AUC, OCR character error rate, MRZ field accuracy, segmentation
IoU/Dice, and face-verification EER. Split by identity or source document before computing metrics.

## Safety boundaries

- Do not commit raw datasets, face images, OCR dumps, or model weights.
- Do not log OCR text, face embeddings, or uploaded documents.
- Require explicit consent before face comparison.
- Treat MRZ checks, model scores, and fusion output as decision support, not definitive proof.
- Do not use face similarity as the sole basis for identity, eligibility, or access decisions.
- Add encryption, retention limits, authentication, and deletion controls before multi-user deployment.

## Current UI limitations

The Streamlit screening tab now accepts a document type, displays MRZ/pipeline results, and marks
video uploads for future frame analysis. It does not yet run video extraction, face detection, or a
trained IDNet checkpoint. Those components should only be enabled after their datasets, licenses,
evaluation results, and model versions are recorded.