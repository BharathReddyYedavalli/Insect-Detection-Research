"""Offline integration tests for the Kaggle IP102 classification ZIP importer."""
from io import BytesIO
from pathlib import Path
import zipfile

import pytest
from PIL import Image
from shared.import_kaggle import import_archive, inspect_archive
from shared.data import dataset_summary, load_splits


def create_zip(path: Path, omit_validation: bool = False):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("classes.txt", "one\ntwo\nthree\n")
        for split in ("train", "val", "test"):
            if omit_validation and split == "val":
                continue
            for cls in range(3):
                name = f"classification/{split}/{cls}/{split}_{cls:03d}.jpg"
                output = BytesIO()
                Image.new("RGB", (20, 20), (cls * 50, 120, 30)).save(output, format="JPEG")
                archive.writestr(name, output.getvalue())


def test_import_and_audit_synthetic_zip(tmp_path):
    archive = tmp_path / "archive.zip"
    create_zip(archive)
    entries, inspection = inspect_archive(archive, num_classes=3)
    assert len(entries) == 9
    assert inspection["split_counts"] == {"train": 3, "val": 3, "test": 3}
    root = tmp_path / "data" / "ip102_v1.1"
    summary = import_archive(archive, root, num_classes=3)
    assert summary["newly_extracted_images"] == 9
    _, splits, offset = load_splits(root, num_classes=3)
    assert offset == 0
    assert splits["test"][1].filename.startswith("test/")
    report, _ = dataset_summary(splits, offset, 3)
    assert report["splits"]["train"]["total"] == 3
    assert (root / "classes.txt").read_text().startswith("one")
    second = import_archive(archive, root, num_classes=3)
    assert second["newly_extracted_images"] == 0
    assert second["existing_size_matched_images"] == 9


def test_missing_validation_split_rejected(tmp_path):
    archive = tmp_path / "bad.zip"
    create_zip(archive, omit_validation=True)
    with pytest.raises(ValueError, match="missing at least one"):
        inspect_archive(archive, num_classes=3)
