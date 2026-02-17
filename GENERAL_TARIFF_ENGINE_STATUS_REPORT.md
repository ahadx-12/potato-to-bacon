# General Tariff Engineer Status Report

Date: 2026-02-17
Scope reviewed: `TEAAS_ROADMAP.md` goal alignment vs current codebase

## Executive Summary

You are **past prototype** and already have a meaningful TEaaS core:
- Unified analysis endpoint exists (`/v1/teaas/analyze`) with baseline classification, mutation testing, and proof chain.
- Batch BOM workflows exist (`/v1/bom/upload` + `/v1/bom/{upload_id}/analyze`, plus async jobs).
- Tenant identity/scoping is present and wired in core API flows.
- Context system supports demo, slice, top-5 chapter fixtures, and a live USITC loader path.

However, for the stated goal of a truly **general** tariff engineer (arbitrary BOM sets, sane savings across broad product space), the main blockers are:
1. Category/fact inference is still mostly keyword and limited to a small enum.
2. Mutation discovery still mixes generalized and legacy heuristic paths.
3. Default context coverage is still a narrow slice unless live USITC data is available.
4. Overlay engine wiring is still mostly 232/301 in runtime path.
5. Portfolio outputs aggregate rates, but enterprise-grade confidence/QA gates are not yet enforced.

---

## Roadmap Alignment (Where You Stand)

### 1) Phase 1 Foundation

**Status: Mostly complete (with cleanup leftovers).**

What is clearly in place:
- API brand/surface is TEaaS-oriented (`CALE Tariff API`) and routers include tariff/proof/law/teaas/auth/jobs/portfolio/bom/pdf paths.
- `/v1/auth/whoami` exists and returns tenant plan/usage metadata.
- Tenant registry + API-key to tenant resolution exists and is consumed by TEaaS/job/BOM flows.

Remaining mismatch vs roadmap cleanup ideal:
- Legacy non-tariff framing remains in top-level docs (`README.md` still physics-first).

### 2) General BOM Intake and Multi-SKU Flow

**Status: Implemented for MVP path, needs hardening.**

What works now:
- BOM parser and upload endpoint with validation, schema mapping, row preview, and skipped-row reporting.
- BOM analyze endpoint queues batch analysis and executes per-item TEaaS analysis in background.
- Separate async jobs endpoint supports up to 500 items and tenant-scoped job listing/polling.

Gaps to "general engine" quality:
- Long-running BOM API test currently times out in this environment (suggesting perf or integration fragility under test harness).
- Batch runner still serializes Z3 globally; throughput scaling is constrained by lock.

### 3) Classification + Optimization Core

**Status: Strong base, but still category-bound in critical places.**

What is solid:
- TEaaS pipeline does context load -> fact compile -> chapter filter -> baseline duty -> overlay evaluation -> mutation search -> optimized result -> proof chain.
- MutationEngine is in use for discovered candidates.

What limits generality today:
- Category inference is keyword-based and routes to a small fixed category set.
- Product schema category enum remains limited (footwear/fastener/electronics/textile/apparel/furniture/other).
- Legacy mutation candidates are still injected alongside derived mutations.
- Material extraction fallback is keyword-only from free text when structured material data is absent.

### 4) Data Coverage (HTS breadth)

**Status: Partial by default, broader if live feed is bootstrapped.**

What exists:
- Context manifests include slice, top-5 chapter fixture, demo, and live USITC context definitions.
- TEaaS endpoint attempts `HTS_US_LIVE` first (when no context is passed), with fallback to default slice.

Generality risk:
- Default context is still `HTS_US_2025_SLICE` (chapters 64/73/85), so out-of-scope BOMs can degrade to weak/no matches when live dataset is absent.

### 5) Overlay/Trade Remedy Realism

**Status: Core framework exists; runtime wiring still narrow.**

What works:
- Overlay engine computes effective duty from base + overlay adders and evaluates by HTS/origin/import matching.

Where it is still narrow:
- Runtime loader currently wires Section 232 sample + Section 301 files in the core overlay loading path.
- AD/CVD and broader FTA overlay data may exist in data files, but is not fully integrated in the primary overlay rule loader path shown.

---

## What Needs To Be Done Next (Priority Order)

## P0 (must-do for "general" positioning)
1. **Make `HTS_US_LIVE` truly default in deployed env**
   - Ensure USITC bootstrap runs in deploy/startup pipeline.
   - Fail loudly (or degraded-mode flag) when only slice context is available.

2. **Replace keyword category gating with evidence-driven typing**
   - Expand product typing beyond fixed enum and allow multi-label category traits.
   - Drive fact generation from parsed BOM structure first; description keywords second.

3. **Remove legacy mutation fallback from production path**
   - Keep only explainable/verified MutationEngine candidates or gate legacy candidates behind a feature flag.

4. **Wire full overlay corpus (AD/CVD + FTA + exclusions) into runtime path**
   - Keep provenance/citation per overlay decision.
   - Add explicit confidence/manual-review flags when remedy matching is ambiguous.

## P1 (scalability + trust)
5. **Scale solver throughput**
   - Replace global lock + thread runner with worker pool/queue strategy and bounded concurrency.

6. **Add portfolio quality gates**
   - Batch-level confidence score, unresolved-fact counts, manual-review reasons.

7. **Create cross-category golden dataset**
   - 100–500 real-ish SKUs across chapters and origin scenarios.
   - Track precision on classification, overlay application, and savings ranking.

## P2 (productization)
8. **Doc and messaging cleanup**
   - Align README/docs with current TEaaS reality (remove physics-first narrative).

9. **Operational observability**
   - Emit per-step latency, context coverage, mutation hit-rate, and false-positive review rates.

---

## Bottom Line

You are around **60-70%** of the way to a credible **general tariff engineer MVP**:
- Plumbing exists (API, batch BOM, proofs, tenancy, context system).
- Core solver loop exists and can produce savings opportunities.

The remaining work is less about basic features and more about:
- **Coverage breadth** (full HTS + overlays),
- **Generalization quality** (better product/fact inference), and
- **Operational robustness** (throughput + confidence controls).

If you execute P0 + P1 above, you can shift from "good demo engine" to "broad importer-grade engine".
