"""Face detection and Eigenfaces recognition pipeline."""

from pathlib import Path
import re

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# Resolve every project path from this script, not from the terminal folder.
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "Faces"
CASCADE_PATH = SCRIPT_DIR / "haarcascade_frontalface_default.xml"
RAW_ARIEL_DIR = DATA_DIR / "ariel_sharon_raw"
ARIEL_OUT_DIR = DATA_DIR / "ariel_sharon"

# Keep the six class names in a fixed order so their integer labels are stable.
CHARACTERS = [
    "ariel_sharon",
    "chris_evans",
    "chris_hemsworth",
    "mark_ruffalo",
    "robert_downey_jr",
    "scarlett_johansson",
]

# The segmented files follow the same person-name + number + .png pattern.
SEGMENT_NAME_PATTERN = "ariel_sharon{}.png"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
OUTPUT_MARKER_NAME = ".eigenfaces-generated"

# All faces must have the same size before PCA builds the Eigenfaces.
IMAGE_SIZE = (100, 100)
TEST_SIZE = 0.3
RANDOM_STATE = 42
N_EIGENFACES = 80

# These values control the Haar face detector.
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5
MIN_SIZE = (30, 30)


def load_cascade():
    """Load and return the Haar cascade classifier."""
    if not hasattr(cv2, "CascadeClassifier"):
        raise RuntimeError(
            "This OpenCV installation does not include CascadeClassifier. "
            "Install a stable OpenCV 4 package, for example: "
            "python3 -m pip install opencv-python==4.12.0.88"
        )

    if not CASCADE_PATH.is_file():
        raise FileNotFoundError(f"Cascade file not found: {CASCADE_PATH}")

    cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
    if cascade.empty():
        raise RuntimeError(f"Could not load cascade file: {CASCADE_PATH}")
    return cascade


def segment_ariel_sharon(
    cascade,
    raw_dir=RAW_ARIEL_DIR,
    output_dir=ARIEL_OUT_DIR,
):
    """Detect the largest face in each Ariel Sharon image and save each crop."""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw Ariel Sharon folder not found: {raw_dir}")

    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Output path is not a directory: {output_dir}")
    if output_dir.exists():
        existing_items = list(output_dir.iterdir())
        marker_path = output_dir / OUTPUT_MARKER_NAME
        if existing_items and not marker_path.is_file():
            raise ValueError(
                f"Output directory is not owned by this pipeline: {output_dir}. "
                "Use an empty directory so existing files are not overwritten."
            )
    else:
        output_dir.mkdir(parents=True)

    marker_path = output_dir / OUTPUT_MARKER_NAME
    marker_path.touch(exist_ok=True)

    raw_files = sorted(
        path
        for path in raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    saved_count = 0
    generated_names = set()

    for raw_path in raw_files:
        image = cv2.imread(str(raw_path))
        if image is None:
            print(f"Could not read, skipped: {raw_path.name}")
            continue
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        detected_faces = cascade.detectMultiScale(
            gray_image,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS,
            minSize=MIN_SIZE,
        )
        if len(detected_faces) == 0:
            print(f"No face detected, skipped: {raw_path.name}")
            continue

        x, y, width, height = max(
            detected_faces,
            key=lambda rectangle: rectangle[2] * rectangle[3],
        )
        face_crop = image[y : y + height, x : x + width]

        saved_count += 1
        output_name = SEGMENT_NAME_PATTERN.format(saved_count)
        generated_names.add(output_name)
        output_path = output_dir / output_name
        if not cv2.imwrite(str(output_path), face_crop):
            raise OSError(f"Could not save segmented face: {output_path}")

    generated_name_pattern = re.compile(r"ariel_sharon\d+\.png")
    for existing_path in output_dir.iterdir():
        is_old_generated_file = (
            existing_path.is_file()
            and generated_name_pattern.fullmatch(existing_path.name)
            and existing_path.name not in generated_names
        )
        if is_old_generated_file:
            existing_path.unlink()

    print(f"Segmented Ariel Sharon faces: {saved_count}")
    return saved_count


def load_dataset(data_dir=DATA_DIR):
    """Load all classes as equalized 100 x 100 grayscale images."""
    data_dir = Path(data_dir)
    face_images = []
    labels = []
    label_to_name = {}

    for label, character in enumerate(CHARACTERS):
        character_dir = data_dir / character
        if not character_dir.is_dir():
            raise FileNotFoundError(f"Character folder not found: {character_dir}")
        label_to_name[label] = character

        image_files = sorted(
            path
            for path in character_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_files:
            raise ValueError(f"No images found in: {character_dir}")

        for image_path in image_files:
            gray_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if gray_image is None:
                print(f"Could not read, skipped: {image_path}")
                continue
            resized_image = cv2.resize(gray_image, IMAGE_SIZE)
            equalized_image = cv2.equalizeHist(resized_image)
            face_images.append(equalized_image)
            labels.append(label)

    faces_array = np.asarray(face_images, dtype=np.uint8)
    labels_array = np.asarray(labels, dtype=np.int32)
    return faces_array, labels_array, label_to_name


def train_and_evaluate(faces, labels, label_to_name):
    """Fit Eigenfaces and an SVM on training data, then return test accuracy."""
    if len(faces) != len(labels):
        raise ValueError("Faces and labels must contain the same number of samples.")
    for label, class_name in label_to_name.items():
        usable_count = int(np.count_nonzero(labels == label))
        if usable_count < 2:
            image_word = "image" if usable_count == 1 else "images"
            raise ValueError(
                f"{class_name} has {usable_count} usable {image_word}; "
                "at least 2 are required for a stratified split."
            )

    flat_faces = faces.reshape(len(faces), -1).astype(np.float32) / 255.0

    train_faces, test_faces, train_labels, test_labels = train_test_split(
        flat_faces,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    component_count = min(
        N_EIGENFACES,
        len(train_faces) - 1,
        train_faces.shape[1],
    )
    eigenface_pca = PCA(
        n_components=component_count,
        whiten=True,
        svd_solver="randomized",
        random_state=RANDOM_STATE,
    )
    train_features = eigenface_pca.fit_transform(train_faces)
    test_features = eigenface_pca.transform(test_faces)

    face_classifier = SVC(C=10.0, kernel="rbf", gamma="scale")
    face_classifier.fit(train_features, train_labels)
    predicted_labels = face_classifier.predict(test_features)

    accuracy = accuracy_score(test_labels, predicted_labels)
    print(f"Training images: {len(train_faces)}")
    print(f"Test images: {len(test_faces)}")
    print(f"Classes: {len(label_to_name)}")
    print(f"Eigenfaces test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    return accuracy


def main():
    """Run segmentation, dataset loading, Eigenfaces training, and evaluation."""
    cascade = load_cascade()
    segment_ariel_sharon(cascade)
    faces, labels, label_to_name = load_dataset()
    print(f"Total usable images: {len(faces)}")
    train_and_evaluate(faces, labels, label_to_name)


if __name__ == "__main__":
    main()
