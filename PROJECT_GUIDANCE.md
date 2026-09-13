# Evidence-First RAG — Project Guidance

> **Purpose:** Keep this file in the repository root. It is a durable handoff note for the project owner, future developers, and AI coding agents. Read it before changing document ingestion, processing, storage, database transactions, retrieval, or production deployment.

## Current baseline

**Project:** Evidence-First RAG for construction-contract and payment-document review.

**Implemented capabilities:**

- Project creation with audit events.
- PDF upload with local-file storage, SHA-256 deduplication per project, document metadata, and audit events.
- Synchronous PDF text extraction with PyMuPDF.
- Whitespace normalization and page-aware chunking.
- Persisted `DocumentPage` records with one-based source page numbers.
- Persisted `Chunk` records linked to their parent page.
- Document lifecycle states: `uploaded`, `processing`, `processed`, and `failed`.
- Audit events for upload, processing start, processing success, and processing failure.
- Clear rejection of PDFs with no extractable text layer; scanned/raster PDFs require a future OCR flow.
- Prevention of duplicate processing once a document is already processed.

**Verified baseline at this milestone:**

```text
python -m pytest -v
10 passed in 63.28s

python -m ruff check app tests
All checks passed!
```
The baseline is updated in the Change log after each verified milestone.

Do not remove or weaken the current unit and integration tests without replacing their coverage intentionally.

## Architecture principles

This system must remain evidence-first.

- Every generated conclusion must be traceable to source evidence.
- Preserve the chain: `Document -> DocumentPage -> Chunk`.
- A future answer/citation model should retain at least `document_id`, `page_id` or `page_number`, and `chunk_id` for every cited claim.
- Do not present unsupported conclusions as facts.
- When source evidence is insufficient, ambiguous, missing, or conflicts, say so and route the case to human review.
- Keep user-facing errors safe and concise; preserve technical detail in controlled logs rather than exposing internals.

## Current processing behavior

Current endpoint:

```text
POST /documents/{document_id}/process
```

Current lifecycle:

```text
uploaded -> processing -> processed
uploaded -> processing -> failed
```

Current HTTP behavior:

| Situation | Status |
|---|---:|
| Document does not exist | 404 |
| Document is already processing | 409 |
| Document is already processed | 409 |
| PDF has no extractable text / needs OCR | 422 |
| File is missing from storage | 422 |
| Database failure while starting or saving processing | 503 |
| Successful extraction and persistence | 200 |

The endpoint is currently **synchronous**: the HTTP request stays open while the application reads the file, extracts text, chunks pages, and writes to the database.

This is acceptable for the current MVP and small text-based PDFs. It is not the long-term processing architecture for large documents, OCR, embeddings, or many simultaneous users.

## Do not change casually

Before changing any item below, add or update tests and explain the migration/rollback plan.

- `DocumentStatus` state semantics.
- Page numbering: page numbers are one-based because users cite PDFs by visible page number.
- `DocumentPage` to `Chunk` relationship and foreign keys.
- SHA-256 duplicate-upload behavior.
- Audit event writes and `trace_id` propagation.
- Error behavior for non-text PDFs requiring OCR.
- Database cleanup and cascade behavior.
- Storage-key conventions.
- Transaction boundaries around document processing.

## Database and transaction rules

- The async SQLAlchemy session may begin an implicit transaction after reads such as `session.get(...)`.
- In the current processing route, use explicit `commit()` and `rollback()` rather than adding `async with session.begin()` after an earlier read on the same session.
- After a failed database operation, call `await session.rollback()` before reusing the session.
- The processing route stores `storage_key` and `project_id` before committing the `processing` transition. Preserve this behavior or deliberately handle SQLAlchemy attribute expiration.
- When recording a processing failure, reload the document by ID instead of relying on a possibly expired ORM object after rollback.
- A successful processing write must persist pages, chunks, final document state, and the processed audit event consistently.

## Engineering lessons and guardrails

### Terminal and API usage

- Python statements such as `from ... import ...` belong in Python files or in `python -c "..."`; do not run them directly as PowerShell commands.
- API route notation such as `GET /documents/{document_id}` is documentation, not a terminal command.
- Exercise API endpoints through integration tests, an HTTP client, Swagger UI, or a running server; do not treat route notation as shell syntax.

### Trace IDs

- `TraceIdMiddleware` accepts `X-Trace-Id` values but truncates them to 64 characters.
- Tests that assert an echoed trace ID must use a value shorter than 64 characters, or explicitly assert the truncated value.
- Preserve trace IDs in request responses, audit events, and safe operational logs.

### SQLAlchemy async sessions

- A read such as `await session.get(...)` can start an implicit transaction.
- Do not add `async with session.begin()` after earlier reads on the same request session unless the transaction state is intentionally managed.
- Use the established explicit `commit()` and `rollback()` pattern in document processing.
- After any `SQLAlchemyError`, call `await session.rollback()` before reusing the session.
- ORM attributes can be expired after `commit()` or `rollback()`. Cache required operational metadata before commits when it will be needed later: `document_id`, `project_id`, `storage_key`, and `size_bytes`.
- When recording a processing failure after rollback, reload the document by `document_id` rather than relying on a previously loaded ORM object.

### PDF and storage behavior

- Minimal bytes starting with `%PDF-` can be sufficient for upload validation tests, but they are not necessarily valid PDFs for text extraction.
- Tests of extraction and processing must generate or use a real text-based PDF, such as one created by PyMuPDF.
- A PDF with no extractable text is not successfully processed. Mark it `failed`, record an audit event, return `422`, and guide the future product flow toward OCR.
- If database metadata exists but the stored file is missing, mark the document `failed`, record `DOCUMENT_FAILED`, return `422`, and do not create pages or chunks.

### Logging and privacy

- Configure application logging centrally through `app/core/logging.py`.
- Use standard-library `logging`; do not introduce a logging dependency without a justified operational requirement.
- Record only safe operational metadata: trace ID, document ID, project ID, size, page count, chunk count, duration, and failure category.
- Never log PDF bytes, extracted page text, chunk text, storage keys, API keys, database URLs, credentials, or customer-sensitive values.
- Keep an automated test that fails if source text or sensitive values appear in processing logs.

### Test workflow

- Run focused unit tests during fast iteration.
- Run the relevant integration test file after route, database, storage, or transaction changes.
- Run `python -m ruff check app tests` and `python -m pytest -v` before a commit, merge, release, or customer deployment.
- Integration tests currently use the real configured database and local storage, so they are slower than unit tests and must always clean up files and rows they create.
- Do not weaken tests merely to make them pass; change a test only when the intended product behavior has changed and is documented.

## Testing rules

Run these before merging meaningful changes:

```powershell
python -m ruff check app tests
python -m pytest -v
```

Current test coverage includes:

- Unit tests for normalization, chunk overlap, invalid chunk settings, valid multi-page extraction, blank PDFs, and invalid PDF bytes.
- Integration tests for project creation and audit events.
- Integration tests for PDF upload, metadata, SHA-256 duplicate prevention, local storage, and audit events.
- Integration tests for successful PDF processing, page/chunk persistence, audit events, and duplicate processing rejection.
- Integration tests for blank PDFs that must become `failed` with no pages/chunks and a failure audit event.

When adding behavior, include failure-path tests, not only happy-path tests.

## Near-term roadmap

Implement in this order unless requirements justify a different order.

1. Add a read endpoint for document status and a bounded, access-controlled way to inspect page/chunk metadata.
2. Add integration coverage for processing an unknown document ID (`404`).
3. Add integration coverage for a file that is uploaded, then missing from storage before processing. Confirm `422`, `failed`, and `DOCUMENT_FAILED`.
4. Add structured logging and timing for processing operations.
5. Enforce upload and processing limits: file size, page count, extraction duration, and maximum chunk count.
6. Add authentication, authorization, tenant isolation, and audit identity before exposing the service to customers.
7. Add retrieval/embedding only with page- and chunk-level citation preservation.
8. Add a separate OCR workflow for scanned/raster documents.
9. Move processing to a background job queue when workload requires it.

## Observability requirements

Before production/customer use, collect and monitor at least:

- `trace_id`
- `project_id`
- `document_id`
- document size in bytes
- page count
- chunk count
- processing duration
- final document status
- processing failure category
- queue wait time and retry count once background jobs exist

Watch for these signals:

| Signal | Likely implication |
|---|---|
| Processing duration rising | Larger or more complex PDFs; capacity is insufficient |
| 502/504/timeouts | Synchronous request exceeds an infrastructure timeout |
| High sustained CPU | Extraction is competing with API request handling |
| Many long-lived `processing` documents | Stuck processing, crashes, or insufficient worker capacity |
| Database pool/connection errors | Too many concurrent database operations or long transactions |
| Growing OCR-required failure rate | Customers upload scanned PDFs; OCR workflow is needed |

## When to use a queue

Do **not** add Redis/Celery/RQ/Dramatiq or another queue merely because it sounds more advanced. A queue adds operational complexity: a broker, workers, monitoring, retry policy, idempotency, deployments, and recovery procedures.

Move to an asynchronous worker architecture when one or more of these are true:

- Customers upload long documents or document packages with many pages.
- OCR is enabled.
- Embedding/indexing is performed after extraction.
- Processing regularly takes more than a few seconds.
- Multiple users process documents at the same time.
- The deployment platform has short request timeouts.
- Progress reporting, retries, and resilient recovery are requirements.

Target queue-based design:

```text
POST /documents/{document_id}/process
  -> validates request
  -> marks document as processing
  -> enqueues an idempotent job
  -> returns 202 Accepted

Worker
  -> reads file
  -> extracts text or performs OCR
  -> writes pages and chunks
  -> writes audit events
  -> marks document processed or failed

GET /documents/{document_id}
  -> returns status, processed_at, error_message, and bounded progress/details
```

The existing status fields and audit design are deliberately useful for this migration. Preserve them.

## Customer safety and commercial readiness

Before selling or hosting for customers, address these controls:

- Authentication and role-based authorization.
- Tenant isolation: ensure one customer cannot access another customer’s documents, chunks, audit events, or storage objects.
- HTTPS in transit and encryption at rest.
- Secrets management; never commit `.env` files, database URLs, API keys, or credentials.
- Upload validation: content type, PDF signature, file size, page count, rate limits, and malware scanning where applicable.
- Resource limits and timeouts to prevent denial-of-service via pathological PDFs.
- Retention and deletion policy for uploaded PDFs, extracted text, embeddings, logs, and backups.
- Audit trails that eventually record actor identity in addition to `trace_id`.
- Backups plus tested restoration procedures.
- Monitoring, alerting, incident response, and a privacy/security contact process.
- Clear product language: the tool assists document review and must not claim legal, contractual, financial, or payment conclusions without cited evidence and human review.

## Guidance for AI coding agents

An AI agent changing this repository must:

1. Read this file, the relevant models, schemas, services, routes, and existing tests before proposing changes.
2. Preserve evidence traceability from result to chunk/page/document.
3. Avoid inventing model fields, enum actions, fixtures, database constraints, or route registration patterns. Inspect the repository first.
4. Prefer small, testable changes over broad rewrites.
5. Add or update tests for each behavior change, including failure paths.
6. Run `python -m ruff check app tests` and the relevant pytest suite before declaring a change complete.
7. Avoid irreversible schema/data changes without an explicit migration and rollback plan.
8. Keep customer data out of logs, test fixtures, prompts, and source control.
9. Explain any change to HTTP semantics, state transitions, audit behavior, error messages, or data-retention behavior.
10. Treat OCR, embeddings, retrieval, LLM output, and background workers as separate reliability and security boundaries.

## Operational checklist before release

- [ ] All tests pass.
- [ ] Ruff passes.
- [ ] Environment variables and secrets are configured outside source control.
- [ ] Database migration status is verified.
- [ ] Storage directory/object storage permissions are restricted.
- [ ] Authentication and authorization are enabled.
- [ ] Tenant isolation is tested.
- [ ] File size, page count, and request-rate limits are enabled.
- [ ] Structured logs and error monitoring are enabled.
- [ ] Backup and restore have been tested.
- [ ] Retention/deletion policy is implemented and documented.
- [ ] OCR-required documents have a deliberate user flow.
- [ ] Customer-facing outputs cite source document and page evidence.
- [ ] Human-review escalation exists for uncertainty or insufficient evidence.

## Change log

### 2026-09-12 — PDF processing milestone

- Added synchronous document processing route.
- Added page-aware PDF extraction and persisted chunks.
- Added processed/failed status handling and audit evidence.
- Added integration tests for successful processing and blank-PDF failure behavior.
- Verified full suite: 10 passing tests.
- Verified Ruff: no findings.


### 2026-09-13 — Document status and safe processing logs

- Added `GET /documents/{document_id}` to return document metadata and current processing status without returning extracted page text or chunks.
- Added integration coverage for `GET` success (`200`) and an unknown document (`404`), including `X-Trace-Id` propagation.
- Added integration coverage for a document whose stored file is missing before processing.
- Added standard-library logging configuration in `app/core/logging.py`, initialized by `app/main.py`.
- Processing logs now record only operational metadata: trace ID, document ID, project ID, size, page/chunk counts, duration, and failure category.
- Processing logs must never include PDF bytes, extracted text, chunks, `storage_key`, API keys, database URLs, or customer-sensitive values.
- Added an automated logging test that verifies source text and sensitive test values do not appear in processing logs.
- Verified baseline after this milestone: `13 passed` and `python -m ruff check app tests` passes.

### 2026-09-13 - Configurable PDF page limit

- Added `MAX_PDF_PAGE_COUNT` with a default value of `250`.
- Page limits are a safety control for the current synchronous processing path, not a permanent product limitation.
- The limit is configurable per deployment through environment variables.
- A document above the configured page limit is rejected with `422`, marked `failed`, records `DOCUMENT_FAILED`, and does not persist pages or chunks.
- Added unit and integration coverage for page-limit rejection.
- Verified baseline after this milestone: `15 passed` and `python -m ruff check app tests` passes.
