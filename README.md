# Eigenfaces Face Recognition

A Python pipeline for face detection and multi-class face recognition using Haar cascades, PCA Eigenfaces, and an SVM classifier.

## Pipeline

1. Detect faces in raw images with OpenCV's Haar cascade.
2. Keep the largest detected face and save the crop.
3. Convert all face images to grayscale, resize them to 100 × 100, and apply histogram equalization.
4. Split the data into training and test sets with a reproducible, stratified 70/30 split.
5. Fit PCA only on the training images. The PCA directions are the Eigenfaces.
6. Train an RBF SVM on the training Eigenface features and evaluate it on held-out images.

Fitting PCA after the split prevents information from the test set from leaking into the learned Eigenface basis.

## Reproduced result

The verified local run produced:

| Metric | Value |
|---|---:|
| Classes | 6 |
| Total usable images | 299 |
| Training images | 209 |
| Test images | 90 |
| Successfully segmented raw faces | 49 of 50 |
| Test accuracy | 83.33% |

The full machine-readable result is in [`results/metrics.json`](results/metrics.json).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Place `haarcascade_frontalface_default.xml` and a `Faces` directory beside `eigenfaces_face_recognition.py`:

```text
eigenfaces-face-recognition/
├── eigenfaces_face_recognition.py
├── haarcascade_frontalface_default.xml
└── Faces/
    ├── ariel_sharon_raw/
    ├── ariel_sharon/        # generated output; start absent or empty
    ├── chris_evans/
    ├── chris_hemsworth/
    ├── mark_ruffalo/
    ├── robert_downey_jr/
    └── scarlett_johansson/
```

Then run:

```bash
python3 eigenfaces_face_recognition.py
```

The segmentation step places a `.eigenfaces-generated` marker in its output directory. On later runs it will update only that pipeline-owned directory. If `Faces/ariel_sharon` already contains files without the marker, the script stops instead of overwriting or deleting them; use a new empty directory for generated crops.

## Dataset note

The face images are not included in this repository. This keeps the repository focused on the implementation and avoids redistributing third-party images. The code expects each identity to have its own directory.

## Tests

```bash
python3 -m pytest -v
```

The tests cover image/label alignment, unreadable files, reproducible PCA/SVM evaluation, PCA component capping, Haar cascade loading, and public project naming.

## Limitations

- Haar cascades may miss faces with difficult poses, lighting, or occlusion. One of the 50 raw images was not detected in the reproduced run.
- Resizing does not align facial landmarks, so pose and crop differences can reduce accuracy.
- The evaluation uses one fixed train/test split. Cross-validation would provide a more stable estimate.
- The dataset is small and is intended for demonstrating the pipeline rather than production identity verification.

## Project summary

Built a reproducible face-recognition workflow in Python using OpenCV, NumPy, scikit-learn, PCA Eigenfaces, and an RBF SVM. The implementation prevents PCA data leakage by fitting the Eigenface basis only on the training split and reached 83.33% accuracy on a six-class, 299-image evaluation.
