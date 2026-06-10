"""Upload generated map images to Cloudinary."""
from __future__ import annotations

import os
from pathlib import Path

import httpx


def upload_unsigned(
    image_path: str | Path,
    *,
    cloud_name: str | None = None,
    upload_preset: str | None = None,
    folder: str | None = None,
) -> str:
    """Upload an image with a Cloudinary unsigned upload preset."""
    resolved_cloud_name = cloud_name or os.getenv("CLOUDINARY_CLOUD_NAME")
    resolved_preset = upload_preset or os.getenv("CLOUDINARY_UPLOAD_PRESET")
    resolved_folder = folder if folder is not None else os.getenv("CLOUDINARY_FOLDER", "tornado-caster")

    if not resolved_cloud_name or not resolved_preset:
        raise ValueError(
            "Set CLOUDINARY_CLOUD_NAME and CLOUDINARY_UPLOAD_PRESET to upload images."
        )

    path = Path(image_path)
    url = f"https://api.cloudinary.com/v1_1/{resolved_cloud_name}/image/upload"
    data = {"upload_preset": resolved_preset}
    if resolved_folder:
        data["folder"] = resolved_folder

    with path.open("rb") as image_file:
        response = httpx.post(
            url,
            data=data,
            files={"file": (path.name, image_file, "image/png")},
            timeout=60.0,
        )
    response.raise_for_status()
    payload = response.json()
    secure_url = payload.get("secure_url")
    if not secure_url:
        raise ValueError("Cloudinary upload succeeded but did not return secure_url.")
    return str(secure_url)
