from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.schemas.evidence import (
    Citation,
    EvidenceSearchRequest,
    EvidenceSearchResponse,
)


def test_citation_accepts_complete_verifiable_source_location() -> None:
    citation = Citation(
        chunk_id=uuid4(),
        document_id=uuid4(),
        document_filename="general-conditions.pdf",
        page_number=14,
        chunk_index=2,
        excerpt="The contractor shall provide written notice before termination.",
    )

    assert citation.document_filename == "general-conditions.pdf"
    assert citation.page_number == 14
    assert citation.chunk_index == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("document_filename", ""),
        ("document_filename", " "),
        ("page_number", 0),
        ("page_number", -1),
        ("chunk_index", -1),
        ("excerpt", ""),
        ("excerpt", " "),
    ],
)
def test_citation_rejects_invalid_verifiable_source_location(
    field: str,
    value: str | int,
) -> None:
    values: dict[str, object] = {
        "chunk_id": uuid4(),
        "document_id": uuid4(),
        "document_filename": "general-conditions.pdf",
        "page_number": 14,
        "chunk_index": 2,
        "excerpt": "The contractor shall provide written notice before termination.",
    }
    values[field] = value

    with pytest.raises(ValidationError):
        Citation(**values)


def test_evidence_search_response_represents_no_evidence() -> None:
    response = EvidenceSearchResponse(
        question="What notice is required before termination?",
        evidence=[],
        no_evidence_found=True,
    )

    assert response.evidence == []
    assert response.no_evidence_found is True


def test_evidence_search_response_rejects_blank_question() -> None:
    with pytest.raises(ValidationError):
        EvidenceSearchResponse(
            question=" ",
            evidence=[],
            no_evidence_found=True,
        )


def test_evidence_search_request_accepts_question_and_default_limit() -> None:
    request = EvidenceSearchRequest(
        question="What notice is required before termination?",
    )

    assert request.question == "What notice is required before termination?"
    assert request.limit == 8


@pytest.mark.parametrize(
    ("question", "limit"),
    [
        ("", 8),
        (" ", 8),
        ("\n\t", 8),
        ("Valid question", 0),
        ("Valid question", 21),
    ],
)
def test_evidence_search_request_rejects_invalid_input(
    question: str,
    limit: int,
) -> None:
    with pytest.raises(ValidationError):
        EvidenceSearchRequest(question=question, limit=limit)