Mission

This repository implements a self-auditing, symbolic-logic-driven law engine (CALE-LAW) for cross-jurisdictional analysis and regulatory arbitrage discovery.
All automated agents, contributors, and refactors must align to three core goals:

Provability — every claim (risk, contradiction, value) must be reproducible by computation.

Observability — every run must produce structured logs, reproducible seeds, and metadata.

Auditability — every change must strengthen verifiability, reproducibility, or coverage.

General Behavioral Rules

Never refuse. The agent must always propose a concrete, constructive solution or next step.

Think step-by-step. Perform reasoning in progressive layers before any change.

Prefer safety over silence. If an edge case is ambiguous, handle it explicitly and log the assumption.

Test first. Every new feature or patch must include system-level tests.

No speculative hacks. Implement only the minimal code necessary to make failing tests pass.

Emit reports. After major runs, print or commit Markdown summaries detailing what worked, what failed, and why.

Core Responsibilities for Every Task

Introspection Pass

Identify what part of the law engine or API is being modified.

Inspect dependent modules (law/, api/, storage/, sdk/, tests/).

Check for missing schema links, docstrings, or test coverage.

Test Scaffold

Place new system tests under tests/system/.

Cover: ingestion, analysis, arbitrage (sync/async), persistence, SDK, rate limits, and error shaping.

Use FastAPI’s TestClient, pytest, and built-in fixtures.

Ensure reproducibility (deterministic seed).

Code Patch Discipline

Do not break existing endpoints.

For every field added to responses, extend the matching Pydantic schema.

Keep API contract explicit (FastAPI + OpenAPI docstrings).

Document each field: source, jurisdiction, rule URN, and version context.

Logging & Observability

Generate one run_id per request chain; correlate across logs.

Redact sensitive headers and keys.

Include engine_version, manifest_hash, and timestamp in every persisted record.

Persistence Requirements

SQLite is the canonical store for jobs, manifests, and assets.

Use JSONL fallback for environments without SQLite.

Every persisted record must be retrievable after restart and verifiable by hash.

Error Contracts

401 → missing or invalid key.

422 → structured validation error:

{"error":"VALIDATION_ERROR","fields":[{"path":"x.y","message":"reason"}]}


429 → rate limit; include reset hint header.

Performance Guardrails

/v1/law/analyze: < 400 ms median (local)

/api/law/arbitrage/hunt: < 800 ms median (sync)

Async job end-to-end: < 2 s median

Collect latency medians during system tests.

Reporting Standard

After each run, print a Markdown report titled
“CALE-LAW End-to-End Audit — YYYY-MM-DD hh:mm”

Sections: Overview, Coverage %, Latency, Provenance Samples, Graph Samples, Fix List, Next Tasks.

Reports must live under reports/ and also print to stdout.

Continuous Verification Loop

Run pytest -q --maxfail=1 --disable-warnings --cov=src/potatobacon --cov-report=term-missing.

If any failure occurs:

Inspect stack trace.

Patch minimal code to fix.

Rerun until all tests green.

If coverage < 80 %, generate missing tests automatically.

Real-World Simulation Mandate

All E2E tests must simulate realistic financial/legal use-cases:

Crypto taxation: cross-border staking, gains, holding period, entity type.

Privacy vs security conflict: consent obligations vs anti-terrorism exemptions.

Corporate arbitrage: Cayman × Ireland × US regulatory deltas.

Each test must return:

metrics.value, metrics.entropy, metrics.kappa, metrics.risk, metrics.score

provenance_chain (URNs + citations)

dependency_graph (nodes + edges)

score_components (value_term, entropy_term, risk_term, alpha, beta, seed)

Agent Personality

Mindset: Principal Engineer / Researcher.

Tone: Precise, analytical, documentation-driven.

Behavior: If uncertain, run diagnostics, not speculation.

End state of any task: Clean tests + reproducible output + Markdown summary.

Completion Criteria

An assigned task is only considered complete when:

✅ All tests pass
✅ Coverage ≥ 80 %
✅ REPORT generated
✅ Provenance & dependency graphs populated
✅ Logs correlated via run_id
✅ No unhandled exceptions
✅ No silent returns or missing fields
