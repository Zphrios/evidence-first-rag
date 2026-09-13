from pathlib import Path
from uuid import UUID

STORAGE_ROOT = Path("storage")


def build_document_storage_key(project_id: UUID, document_id: UUID) -> str:
    """Return the relative local-storage key for a PDF document."""
    return f"projects/{project_id}/documents/{document_id}.pdf"


def save_document_bytes(storage_key: str, content: bytes) -> None:
    """Persist PDF bytes under the configured local development storage root."""
    destination = STORAGE_ROOT / storage_key
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)


def delete_document_bytes(storage_key: str) -> None:
    """Remove a stored file when it exists."""
    destination = STORAGE_ROOT / storage_key
    if destination.exists():
        destination.unlink()


def read_document_bytes(storage_key: str) -> bytes:
    """Read stored PDF bytes from the local development storage root."""
    source = STORAGE_ROOT / storage_key

    if not source.is_file():
        raise FileNotFoundError(f"Stored document file not found: {storage_key}")

    return source.read_bytes()