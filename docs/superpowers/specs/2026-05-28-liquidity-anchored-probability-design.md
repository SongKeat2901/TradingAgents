# Liquidity-Anchored 12-Month Probability — Design

**Date:** 2026-05-28 · **Status:** Approved design (not yet implemented) · **Author:** brainstorming session

## Problem

Three weaknesses in the current technical/probabilistic layer:

1. **Volume zones are LLM-eyeballed.** The TA Agent writes a "## Volume profile zones" section in `technicals.md` by estimating ("heaviest accumulation $X–$Y") — no actual volume-by-price computation. The "high volume / most-transacted / liquidity" areas the desk cares about are not measured.
2. **Breakout/breakdown is MA-only.** `classifier.py`'s BREAKOUT/BREAKDOWN classes trigger off MA-gaps + volume + ATR moves, with no awareness of the liquidity levels price is actually breaking through.
3. **Scenario probabilities are guessed.** The Portfolio Manager's Bull/Base/Bear probabilities are LLM-judged, not derived from any analysis of the stock's own history.

This violates the system's core principle — *load-bearing numbers are computed in Python from ground truth, the LLM reasons around them* — for exactly the numbers most central to a trading decision (where the levels are, and how likely they are to be reached).

## Goal

Replace all three with deterministic computation that **hard-anchors** the agents, following the established pattern (classifier, peer_ratios, Phase 7.15 PM injection):

> Volume profile defines **WHERE** the levels are; block-bootstrap Monte Carlo defines **HOW LIKELY** they are to be reached over the next 12 months. Both are computed in Python and injected verbatim.

## Design

### Component 1 — Volume Profile (`tradingagents/agents/utils/volume_profile.py`)

**Input:** `prices.json` OHLCV, two windows — **36-month structural** and **6-month tactical**.

**Method:** volume-by-price histogram. For each daily bar, distribute its volume across the bins spanning its high–low range (uniform distribution across the range; bin width derived from price range / N bins, N tunable, default ~50). Aggregate across all bars in each window.

**Extracted levels (per window):**
- **POC** (Point of Control) — highest-volume price bin (the "most visited area")
- **Value Area** — the contiguous ±range around POC containing 70% of total volume → **VAH** (value-area high) and **VAL** (value-area low) (the "highly transacted zone")
- **High Volume Nodes (HVN)** — local volume-histogram peaks above a prominence threshold → support/resistance magnets (the "high volume / liquidity areas")
- **Low Volume Nodes (LVN)** — local volume troughs / gaps → fast-move "slip-through" zones

**Output:**
- `raw/volume_profile.json` — both windows, each with POC, VAH, VAL, ranked HVN list, LVN list, and per-level volume share.
- A rendered "## Liquidity / Volume profile" Markdown table appended to `pm_brief.md`.
- **Replaces** the LLM-eyeballed "Volume profile zones" section in the TA Agent prompt: the TA Agents receive the computed levels as a verbatim block (same mechanism as the classification block).

### Component 2 — Breakout/Breakdown augmentation (extend `classifier.py`)

**Augment** (not replace) the existing BREAKOUT/BREAKDOWN rules with volume-node confirmation:
- **Breakout** — existing MA/volume/ATR trigger AND price has cleared a major HVN or the VAH on above-average volume; record the specific level cleared + a confirmation flag.
- **Breakdown** — existing trigger AND price has lost the VAL or a major HVN below; record the level + flag.

The MA-based logic is preserved; the volume profile adds the *specific level* context and a confirmation boolean. `classification.json` gains fields: `broken_level`, `broken_level_type` (HVN/VAH/VAL), `volume_confirmed` (bool).

### Component 3 — 12-Month Forward Distribution (`tradingagents/agents/utils/forward_distribution.py`)

**Input:** 36-month daily returns derived from `prices.json`.

**Method — block-bootstrap Monte Carlo:**
- Resample contiguous blocks of actual daily log-returns (block length ~5–20 trading days, to preserve short-horizon autocorrelation and volatility clustering) to assemble ~10,000 simulated forward paths of 252 trading days each.
- Seed the RNG deterministically (e.g. from ticker+trade_date hash) so the same inputs yield the same probabilities — consistency requirement, mirrors the classifier's determinism rationale.

**Targets = volume-profile levels:**
- **Bull target** = next major HVN above spot (fallback: VAH, then a high quantile)
- **Base target** = POC / current acceptance (fallback: spot)
- **Bear target** = next major HVN below spot (fallback: VAL, then a low quantile)

**Probability per target = TOUCH probability:** fraction of simulated paths whose intraday path reaches/closes beyond the level at any point within 12 months. (Terminal-quantile probabilities are also computed and stored as a cross-check, but the anchored Bull/Base/Bear probabilities use touch.)

**Output:** `raw/forward_probabilities.json` — Bull/Base/Bear `{target, touch_prob, terminal_prob}`, plus terminal quantiles (p5–p95) and the simulation parameters (block length, n_paths, seed, realized drift/vol). Probabilities normalized so the three scenarios sum to 100%.

### Integration

- **Researcher node** (`researcher.py`) computes all three after the price fetch, writes the three JSON files, and appends the two deterministic blocks (liquidity levels + forward-probability table) to `pm_brief.md`.
- **TA Agents** receive the volume-profile levels as a verbatim injected block; their "Volume profile zones" section is now sourced from it, not eyeballed.
- **Portfolio Manager** receives the forward-probability block injected directly into its prompt (same mechanism as the Phase 7.15 peer-ratios injection) and must use the Bull/Base/Bear targets + probabilities verbatim.

### Validation

- **New validator** `validators/scenario_probability_validator.py`: checks that the PM's scenario targets and probabilities match `forward_probabilities.json` within tolerance, and that the three probabilities sum to 100%. MATERIAL on mismatch (gates Telegram, like the other Phase 7.x validators).
- Extends `cli/research_validation.py` orchestration + `validation_report.json` schema (`phase_7_x_scenario_probability`).

### Testing

- `volume_profile`: known synthetic OHLCV → asserted POC / VAH / VAL / HVN positions.
- `forward_distribution`: seeded RNG → stable touch probabilities; degenerate inputs (flat price) → sane edges; probabilities sum to 100%.
- `classifier` augmentation: fixtures where price has/hasn't cleared an HVN → correct `volume_confirmed` flag.
- `scenario_probability_validator`: matching and mismatching PM outputs.

## Non-goals (YAGNI)

- No regime-conditional / cross-ticker historical conditioning (considered, deferred — data-hungry).
- No intraday/tick volume profile — daily OHLCV distribution is sufficient.
- No new charting/visualization — JSON + Markdown tables only (the PDF can render tables).
- Subject-ticker forward P/E (separate backlog item) is unrelated and out of scope here.

## Open parameters to tune during implementation

- Histogram bin count (default ~50), HVN prominence threshold, value-area %, block length, n_paths. All defaults stated above; revisit against real runs.

## Rollout

Phased (each independently shippable + testable):
1. Volume Profile module + JSON + pm_brief block + TA injection.
2. Classifier breakout/breakdown augmentation (depends on 1).
3. Forward distribution module + JSON + PM injection (depends on 1 for targets).
4. Scenario-probability validator + delivery gating (depends on 3).
