from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Tuple

from fastapi import UploadFile

from ..config import settings
from ..models.schemas import StoredFileInfo


VIDEO_MIME_PREFIXES = ("video/",)


async def save_upload_file(file_id: str, upload: UploadFile) -> StoredFileInfo:
    storage_dir: Path = settings.storage_root
    storage_dir.mkdir(parents=True, exist_ok=True)

    # Figure out an extension from the filename or content type
    original_name: str = upload.filename or "uploaded"
    guessed_ext = Path(original_name).suffix
    if not guessed_ext:
        # Try to guess from content type
        if upload.content_type:
            extension = mimetypes.guess_extension(upload.content_type) or ""
        else:
            extension = ""
    else:
        extension = guessed_ext

    stored_filename = f"{file_id}{extension}"
    target_path = storage_dir / stored_filename

    with open(target_path, "wb") as out_file:
        content = await upload.read()
        out_file.write(content)

    is_video = (upload.content_type or "").startswith(VIDEO_MIME_PREFIXES)

    public_url: str | None = None
    if settings.public_base_url:
        public_url = f"{settings.public_base_url.rstrip('/')}/{stored_filename}"

    return StoredFileInfo(
        stored_filename=stored_filename,
        local_path=str(target_path),
        public_url=public_url,
        is_video=is_video,
    )




