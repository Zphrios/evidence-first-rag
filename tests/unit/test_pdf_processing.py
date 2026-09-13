import pymupdf
import pytest

from app.services.pdf_processing import (
    ExtractedChunk,
    chunk_page_text,
    extract_and_chunk_pdf,
    normalize_whitespace,
)


def build_pdf_bytes(page_texts: list[str]) -> bytes:
    """Create an in-memory text PDF for deterministic extraction tests."""
    pdf_document = pymupdf.open()

    try:
        for page_text in page_texts:
            page = pdf_document.new_page()
            page.insert_text((72, 72), page_text)

        return pdf_document.write()
    finally:
        pdf_document.close()


def test_normalize_whitespace() -> None:
    """Whitespace normalization preserves words while collapsing gaps."""
    assert normalize_whitespace("Clause  1.1\n\n  Notice\tperiod") == "Clause 1.1 Notice period"


def test_chunk_page_text_preserves_words_and_overlap() -> None:
    """Chunks remain readable and repeat the configured overlap word."""
    chunks = chunk_page_text(
        "alpha beta gamma",
        chunk_size_chars=10,
        overlap_words=1,
    )

    assert chunks == [
        ExtractedChunk(chunk_index=0, text_content="alpha beta"),
        ExtractedChunk(chunk_index=1, text_content="beta gamma"),
    ]


def test_chunk_page_text_rejects_invalid_settings() -> None:
    """Invalid chunk settings fail with clear errors."""
    with pytest.raises(ValueError, match="greater than zero"):
        chunk_page_text("text", chunk_size_chars=0)

    with pytest.raises(ValueError, match="non-negative"):
        chunk_page_text("text", overlap_words=-1)


def test_extract_and_chunk_pdf_preserves_page_numbers() -> None:
    """Extraction retains one-based PDF page numbers and page-local chunks."""
    pdf_bytes = build_pdf_bytes(
        [
            "Page one: scope of work.",
            "Page two: written notice is required within seven days.",
        ]
    )

    pages = extract_and_chunk_pdf(pdf_bytes)

    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert pages[0].text_content == "Page one: scope of work."
    assert pages[0].chunks == [
        ExtractedChunk(
            chunk_index=0,
            text_content="Page one: scope of work.",
        )
    ]

    assert pages[1].page_number == 2
    assert "written notice is required within seven days" in pages[1].text_content
    assert pages[1].chunks[0].chunk_index == 0


def test_extract_and_chunk_pdf_rejects_blank_pdf() -> None:
    """A PDF with no extractable text reports that OCR is required."""
    pdf_bytes = build_pdf_bytes([""])

    with pytest.raises(ValueError, match="no extractable text layer"):
        extract_and_chunk_pdf(pdf_bytes)


def test_extract_and_chunk_pdf_rejects_invalid_bytes() -> None:
    """Non-PDF bytes fail with an explicit parsing error."""
    with pytest.raises(ValueError, match="Invalid or corrupted PDF"):
        extract_and_chunk_pdf(b"not a PDF")


def test_extract_and_chunk_pdf_rejects_page_count_above_limit() -> None:
    """PDFs above the configured page limit are rejected before extraction."""
    pdf_document = pymupdf.open()

    for page_number in range(3):
        page = pdf_document.new_page()
        page.insert_text(
            (72, 72),
            f"Page {page_number + 1} contains extractable contract text.",
        )

    pdf_bytes = pdf_document.tobytes()
    pdf_document.close()

    with pytest.raises(
        ValueError,
        match="PDF page count exceeds the configured limit of 2.",
    ):
        extract_and_chunk_pdf(pdf_bytes, max_page_count=2)
        
        
def test_extract_and_chunk_pdf_rejects_chunk_count_above_limit() -> None:
    """PDFs above the configured chunk limit are rejected before persistence."""
    pdf_bytes = build_pdf_bytes(
        [
            " ".join(f"page_one_word_{index}" for index in range(200)),
            " ".join(f"page_two_word_{index}" for index in range(200)),
        ]
    )

    with pytest.raises(
        ValueError,
        match="PDF chunk count exceeds the configured limit of 1.",
    ):
        extract_and_chunk_pdf(pdf_bytes, max_chunk_count=1)