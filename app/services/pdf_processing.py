from dataclasses import dataclass

import pymupdf

CHUNK_SIZE_CHARS = 1_200
CHUNK_OVERLAP_WORDS = 25


@dataclass(frozen=True)
class ExtractedChunk:
    """A readable chunk of one page, indexed from zero within that page."""

    chunk_index: int
    text_content: str


@dataclass(frozen=True)
class ExtractedPage:
    """Extracted text and page-local chunks using a one-based page number."""

    page_number: int
    text_content: str
    chunks: list[ExtractedChunk]


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace without changing the sequence of non-whitespace characters."""
    return " ".join(text.split())


def chunk_page_text(
    text: str,
    chunk_size_chars: int = CHUNK_SIZE_CHARS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list[ExtractedChunk]:
    """Split one page into readable chunks with a word-based overlap."""
    if chunk_size_chars <= 0:
        raise ValueError("chunk_size_chars must be greater than zero.")

    if overlap_words < 0:
        raise ValueError("overlap_words must be non-negative.")

    normalized_text = normalize_whitespace(text)
    if not normalized_text:
        return []

    words = normalized_text.split(" ")
    chunks: list[ExtractedChunk] = []
    start_word_index = 0
    chunk_index = 0

    while start_word_index < len(words):
        end_word_index = start_word_index
        current_length = 0

        while end_word_index < len(words):
            word = words[end_word_index]
            separator_length = 1 if end_word_index > start_word_index else 0
            next_length = current_length + separator_length + len(word)

            if current_length and next_length > chunk_size_chars:
                break

            current_length = next_length
            end_word_index += 1

        if end_word_index == start_word_index:
            end_word_index += 1

        chunk_text = " ".join(words[start_word_index:end_word_index])
        chunks.append(
            ExtractedChunk(
                chunk_index=chunk_index,
                text_content=chunk_text,
            )
        )
        chunk_index += 1

        if end_word_index >= len(words):
            break

        next_start_word_index = max(
            end_word_index - overlap_words,
            start_word_index + 1,
        )
        start_word_index = next_start_word_index

    return chunks


def extract_and_chunk_pdf(
    pdf_bytes: bytes,
    max_page_count: int | None = None,
    max_chunk_count: int | None = None,
) -> list[ExtractedPage]:
    """Extract text and page-local chunks from a text-based PDF payload."""
    if not pdf_bytes:
        raise ValueError("Cannot extract an empty PDF payload.")

    if max_page_count is not None and max_page_count <= 0:
        raise ValueError("max_page_count must be greater than zero when provided.")
    
    if max_chunk_count is not None and max_chunk_count <= 0:
        raise ValueError("max_chunk_count must be greater than zero when provided.")

    try:
        pdf_document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError("Invalid or corrupted PDF file.") from exc

    try:
        if pdf_document.page_count == 0:
            raise ValueError("PDF contains zero pages.")

        if (
            max_page_count is not None
            and pdf_document.page_count > max_page_count
        ):
            raise ValueError(
                f"PDF page count exceeds the configured limit of {max_page_count}."
            )

        extracted_pages: list[ExtractedPage] = []
        total_chunk_count = 0
        total_text_length = 0

        for page_number, page in enumerate(pdf_document, start=1):
            text_content = normalize_whitespace(page.get_text("text"))
            page_chunks = chunk_page_text(text_content)

            total_text_length += len(text_content)
            total_chunk_count += len(page_chunks)

            if max_chunk_count is not None and total_chunk_count > max_chunk_count:
                raise ValueError(
                    f"PDF chunk count exceeds the configured limit of {max_chunk_count}."
                )

            extracted_pages.append(
                ExtractedPage(
                    page_number=page_number,
                    text_content=text_content,
                    chunks=page_chunks,
                )
            )    
    finally:
        pdf_document.close()

    if total_text_length == 0:
        raise ValueError(
            "PDF contains no extractable text layer (scanned/raster document requires OCR)."
        )

    return extracted_pages