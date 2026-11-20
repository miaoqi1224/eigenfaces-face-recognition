from pathlib import Path

import cv2
import numpy as np
import pytest

import eigenfaces_face_recognition as pipeline


def write_gray_image(path: Path, value: int) -> None:
    image = np.full((12, 12), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_load_dataset_keeps_images_and_labels_aligned(tmp_path, monkeypatch):
    """Removing the per-class label append must make this test fail."""
    monkeypatch.setattr(pipeline, "CHARACTERS", ["class_a", "class_b"])
    for class_name, values in {"class_a": [10, 20], "class_b": [220, 230]}.items():
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for index, value in enumerate(values):
            write_gray_image(class_dir / f"image_{index}.png", value)

    faces, labels, label_to_name = pipeline.load_dataset(tmp_path)

    assert faces.shape == (4, 100, 100)
    assert labels.tolist() == [0, 0, 1, 1]
    assert label_to_name == {0: "class_a", 1: "class_b"}


def test_load_dataset_skips_unreadable_images_without_extra_labels(tmp_path, monkeypatch):
    """Appending a label before confirming image readability must fail here."""
    monkeypatch.setattr(pipeline, "CHARACTERS", ["class_a", "class_b"])
    for class_name, value in (("class_a", 30), ("class_b", 210)):
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        write_gray_image(class_dir / "valid.png", value)
        (class_dir / "broken.png").write_bytes(b"not an image")

    faces, labels, _ = pipeline.load_dataset(tmp_path)

    assert len(faces) == 2
    assert labels.tolist() == [0, 1]


def test_train_and_evaluate_caps_pca_and_is_deterministic(monkeypatch):
    """Removing PCA capping or the fixed split seed must make this test fail."""
    rng = np.random.default_rng(7)
    class_zero = rng.integers(0, 40, size=(12, 4, 4), dtype=np.uint8)
    class_one = rng.integers(215, 255, size=(12, 4, 4), dtype=np.uint8)
    faces = np.concatenate([class_zero, class_one])
    labels = np.array([0] * 12 + [1] * 12, dtype=np.int32)
    monkeypatch.setattr(pipeline, "N_EIGENFACES", 80)

    first = pipeline.train_and_evaluate(faces, labels, {0: "dark", 1: "light"})
    second = pipeline.train_and_evaluate(faces, labels, {0: "dark", 1: "light"})

    assert first == second
    assert 0.0 <= first <= 1.0


def test_train_and_evaluate_caps_pca_by_feature_width(monkeypatch):
    """Capping PCA only by sample count must make this test fail."""
    rng = np.random.default_rng(11)
    faces = rng.integers(0, 255, size=(24, 2, 2), dtype=np.uint8)
    labels = np.array([0] * 12 + [1] * 12, dtype=np.int32)
    monkeypatch.setattr(pipeline, "N_EIGENFACES", 80)

    score = pipeline.train_and_evaluate(faces, labels, {0: "a", 1: "b"})

    assert 0.0 <= score <= 1.0


def test_train_and_evaluate_rejects_missing_expected_class():
    """Letting a registered class contribute zero samples must fail here."""
    faces = np.zeros((4, 4, 4), dtype=np.uint8)
    labels = np.zeros(4, dtype=np.int32)

    with pytest.raises(ValueError, match=r"class_b.*0 usable images"):
        pipeline.train_and_evaluate(faces, labels, {0: "class_a", 1: "class_b"})


def test_train_and_evaluate_rejects_singleton_before_split():
    """Delegating singleton validation to scikit-learn must fail here."""
    faces = np.zeros((5, 4, 4), dtype=np.uint8)
    labels = np.array([0, 0, 0, 0, 1], dtype=np.int32)

    with pytest.raises(ValueError, match=r"class_b.*1 usable image"):
        pipeline.train_and_evaluate(faces, labels, {0: "class_a", 1: "class_b"})


class FixedFaceCascade:
    def detectMultiScale(self, image, **kwargs):
        return np.array([[1, 1, 8, 8]], dtype=np.int32)


def test_segmentation_marks_new_output_as_pipeline_owned(tmp_path):
    """Creating generated crops without an ownership marker must fail here."""
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "generated"
    raw_dir.mkdir()
    color_image = np.full((12, 12, 3), 128, dtype=np.uint8)
    assert cv2.imwrite(str(raw_dir / "face.png"), color_image)

    first_count = pipeline.segment_ariel_sharon(
        FixedFaceCascade(), raw_dir=raw_dir, output_dir=output_dir
    )
    second_count = pipeline.segment_ariel_sharon(
        FixedFaceCascade(), raw_dir=raw_dir, output_dir=output_dir
    )

    assert first_count == second_count == 1
    assert (output_dir / ".eigenfaces-generated").is_file()


def test_segmentation_rejects_unowned_nonempty_output(tmp_path):
    """Deleting a pre-existing crop from an unowned directory must fail here."""
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "existing"
    raw_dir.mkdir()
    output_dir.mkdir()
    existing_crop = output_dir / "ariel_sharon1.png"
    existing_crop.write_bytes(b"user supplied")

    with pytest.raises(ValueError, match="not owned by this pipeline"):
        pipeline.segment_ariel_sharon(
            FixedFaceCascade(), raw_dir=raw_dir, output_dir=output_dir
        )

    assert existing_crop.read_bytes() == b"user supplied"


def test_load_cascade_uses_configured_xml(monkeypatch):
    """Ignoring the configured cascade path must make this test fail."""
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    monkeypatch.setattr(pipeline, "CASCADE_PATH", cascade_path)

    cascade = pipeline.load_cascade()

    assert not cascade.empty()


def test_public_module_title_is_portfolio_facing():
    """Changing the public project title must make this test fail."""
    assert pipeline.__doc__ == "Face detection and Eigenfaces recognition pipeline."
