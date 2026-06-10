from pathlib import Path

import pytest

from src.publishing.cloudinary_upload import upload_unsigned


def test_cloudinary_upload_requires_cloud_name_and_preset(monkeypatch, tmp_path):
    monkeypatch.delenv("CLOUDINARY_CLOUD_NAME", raising=False)
    monkeypatch.delenv("CLOUDINARY_UPLOAD_PRESET", raising=False)
    image = tmp_path / "image.png"
    image.write_bytes(b"not really a png")

    with pytest.raises(ValueError):
        upload_unsigned(image)


def test_cloudinary_upload_requires_secure_url(monkeypatch, tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"not really a png")

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {}

    def fake_post(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr("src.publishing.cloudinary_upload.httpx.post", fake_post)

    with pytest.raises(ValueError):
        upload_unsigned(
            Path(image),
            cloud_name="demo",
            upload_preset="unsigned_preset",
        )
