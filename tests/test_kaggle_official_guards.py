"""Regression test: nonbenchmark split counts cannot silently pass as official IP102."""
from io import BytesIO
import zipfile

import pytest
from PIL import Image

from shared.import_kaggle import inspect_archive, import_archive


def test_refuse_nonstandard_102_class_archive(tmp_path):
    archive = tmp_path / "unofficial.zip"
    image = BytesIO()
    Image.new("RGB", (3, 3), (20, 40, 60)).save(image, format="JPEG")
    with zipfile.ZipFile(archive, "w") as zf:
        for label in range(102):
            zf.writestr(f"classification/train/{label}/a.jpg", image.getvalue())
        zf.writestr("classification/val/0/b.jpg", image.getvalue())
        zf.writestr("classification/test/0/c.jpg", image.getvalue())
    _, summary = inspect_archive(archive, 102)
    assert summary["images_total"] == 104
    assert not summary["official_split_counts_match"]
    with pytest.raises(ValueError, match="differ from the published IP102 benchmark"):
        import_archive(archive, tmp_path / "data")
    assert not (tmp_path / "data").exists()
