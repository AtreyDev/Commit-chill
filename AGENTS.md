# VeriLogic Project Brief

## Product goal
Build VeriLogic as an SIH26188 AI-based fake identity and document screening product. The visible product must connect:

- OCR and MRZ extraction;
- deterministic document-field and checksum validation;
- forensic tamper detection with explainable evidence;
- optional face verification as decision support, never sole proof;
- tamper-evident audit records and human-review routing.

## Current architecture
- Python 3.11+ and Streamlit are the current hackathon runtime.
- `app.py` is the entry point.
- `src/screening.py`, `src/logic.py`, `src/reconciliation.py`, and `src/audit.py` power VeriLogic.
- Existing document-forensics modules must remain available and working.
- The optional wellness dashboard is legacy code and must remain clearly secondary to document screening.

## Engineering rules
- Keep the app offline-first and deterministic where possible.
- Never hard-code, print, persist, or render GitHub tokens or other secrets.
- Burnout output is a workload heuristic, never a diagnosis or employee-surveillance claim.
- Prefer transparent rules and evidence over generic AI features.
- Reuse existing modules; avoid unnecessary rewrites.
- Add focused tests for every new service and run them before broad changes.
- Do not add a Node/Next.js frontend to the Python root unless a deliberate migration begins in a separate folder.
- Do not commit generated secrets, `.env`, `.docauth/`, uploads, model weights, or local caches.

## High-value roadmap
1. Improve GitHub OAuth/API coverage and rate-limit handling.
2. Add privacy-preserving team aggregates only after a real backend/auth boundary exists.
3. Add calendar integration and scheduled notifications behind explicit consent.
4. Consider a separate `frontend/` Next.js + FastAPI migration only when deployment needs it.

## Validation
- `uv run --with pytest pytest tests/test_commit_chill.py -q`
- `uv run --with pytest pytest tests/test_verilogic.py tests/test_screening.py tests/test_analysis.py tests/test_copy_move.py -q`
- `uv run streamlit run app.py`
