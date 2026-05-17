# Implementation Decision: Easiest + Most Effective Path

## Decision

Use a **hybrid parity-first path**:

1. Keep Python as the ingestion, feature, orchestration, and LLM layer.
2. Treat Nadocast upstream as source-of-truth for:
   - canonical feature order (`2005`-feature list)
   - model routing (event + forecast window -> model path)
3. For model inference parity, use Nadocast-native `MemoryConstrainedTreeBoosting` format via a Julia sidecar when available.

## Why this is best

- Lowest risk for exact parity: avoids brittle reverse-engineering of binary `.model` format.
- Fastest practical path: we can continue building Python pipeline now while inference bridge is isolated.
- Scales cleanly: once Julia sidecar is available, same contract + same features plug into production.

## What is implemented now

- Contract extraction from upstream:
  - `artifacts/upstream/model_contract.json`
  - `artifacts/upstream/features_order_2005.txt`
- Optional single-model export (no huge pull)
- Binary model guardrail in Python bridge
- Contract-aware model resolution in `run_pipeline(...)`

## Execution order (top-to-bottom plan alignment)

1. Step 1 Ingestion: keep hardening async byte-range pulls and variable selectors.
2. Step 2 Features: extend Python features from current blocks toward full upstream feature semantics in feature-order file.
3. Step 3 Inference:
   - short term: mock LightGBM for pipeline continuity
   - parity path: Julia sidecar for native `.model` evaluation
4. Step 4 LLM: derive top-driving features from actual inference outputs once Step 3 parity path is active.
5. Step 5 Orchestration: daemonize and add memory-bounded end-to-end jobs.

