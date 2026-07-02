# FA-101 WP5a — Short Interest + Analyst Consensus Block

**Date:** 2026-07-02
**Status:** Approved (self-approved under the standing FA-101 alignment goal), pending plan
**Scope:** Close FA-101 §8 (Catalysts & Sentiment) deterministic gaps: short interest and analyst consensus / price targets, as a deterministic `pm_brief.md` block. First of the WP5 (new-free-data) phases; the cheap high-signal one — all fields are free in `ticker.info`, already fetched by `get_fundamentals` (no extra network call).

---

## Background

FA-101 §8 flagged missing **short interest**, **analyst consensus & revisions**, and **price targets**. Live check confirmed yfinance `ticker.info` carries all of them, populated for both large- and small-caps: `sharesShort`, `sharesShortPriorMonth`, `shortRatio` (days-to-cover), `shortPercentOfFloat`; `recommendationKey`, `recommendationMean`, `numberOfAnalystOpinions`, `targetMeanPrice`/`targetMedianPrice`/`targetHighPrice`/`targetLowPrice`, `currentPrice`. `get_fundamentals` (`dataflows/y_finance.py`) already calls `ticker.info`, so these are added to its field-list with zero extra fetch.

## Design principles

- **Deterministic-block pattern:** add the fields to the `get_fundamentals` blob (no new call) → a new block computes/formats → append to `pm_brief.md` → the LLM cites verbatim.
- **Free-data honesty:** any field absent → that metric renders `n/a (data unavailable)`; smaller/foreign names may be sparse. Never fabricate.
- **Clear role (aligns with the multi-agent goal):** sentiment/consensus is the **social analyst's** domain — mandate the citation there (it already owns sentiment §8).
- **No new QC item.**

## Part 1 — surface the fields (`dataflows/y_finance.py get_fundamentals`)

Add these `("Label", info.get(key))` pairs to the existing field list (info is already fetched):
`Shares Short` (sharesShort), `Shares Short Prior Month` (sharesShortPriorMonth), `Short Ratio Days To Cover` (shortRatio), `Short Percent Of Float` (shortPercentOfFloat), `Analyst Recommendation` (recommendationKey), `Analyst Recommendation Mean` (recommendationMean), `Number Of Analyst Opinions` (numberOfAnalystOpinions), `Target Mean Price` (targetMeanPrice), `Target Median Price` (targetMedianPrice), `Target High Price` (targetHighPrice), `Target Low Price` (targetLowPrice). (`Current Price` / `Market Cap` etc. are already present.) The existing `if value is not None` filter drops absent fields cleanly.

## Part 2 — block module `agents/utils/sentiment_consensus.py`

`compute_sentiment_consensus(financials: dict) -> dict` parses the fields from `financials["fundamentals"]` (regex, the established `_num`/`_text` pattern used by `intrinsic_value.parse_fundamentals`), computes:
- **Short interest:** `short_pct_float` = shortPercentOfFloat×100; `days_to_cover` = shortRatio; `short_mom_change_pct` = (sharesShort−sharesShortPriorMonth)/sharesShortPriorMonth×100 (rising/falling).
- **Consensus:** `rating` (recommendationKey, e.g. "strong_buy"), `rating_mean` (recommendationMean, 1=buy…5=sell), `n_analysts`.
- **Price target:** `target_mean`, `target_median`, `target_upside_pct` = (target_mean/current−1)×100 (uses the reference price if current absent), `target_low`/`target_high` range.

`format_sentiment_block(result) -> str` → `## Sentiment & consensus (short interest + analyst view)` table; each missing metric → `n/a (data unavailable)`; a caveat line (short interest & targets are point-in-time yfinance snapshots). Empty/unparseable → an honest "unavailable" one-liner.

## Part 3 — wiring & citation

- `researcher.py`: after an existing block, `sc = compute_sentiment_consensus(financials)` → write `raw/sentiment_consensus.json` → append `format_sentiment_block(sc)` to `pm_brief.md`, own try/except (fail-open).
- Citation: mandate the **social analyst** (`agents/analysts/social_media_analyst.py`) cite the short-interest %/days-to-cover and consensus rating + target upside verbatim from the block, when present. (Confirm the social analyst receives `pm_brief.md` via its `format_for_prompt(files=[...])`; if not, add it.) No new QC item.

## Testing

- **Block (`tests/test_sentiment_consensus.py`):** a fundamentals-blob fixture with the fields → correct short_pct_float / days_to_cover / short_mom_change_pct / rating / target_upside_pct (hand-computed); missing fields → the affected metrics `None`; empty blob → "unavailable"; block renders heading + a value + the caveat.
- **Prompt (`tests/test_social_prompt.py` or existing):** the social analyst `_SYSTEM` mandates citing the sentiment/consensus block (distinctive substring).
- **Real success:** on live tickers, sane values (MSFT ≈ 1.3% float / strong_buy / ~+46% target; sparse names degrade to n/a).

## Out of scope (later WP5 phases / program)

- 13F institutional ownership, 13D/G activist stakes, 8-K, DEF 14A proxy (need SEC fetch) — later WP5 phases.
- Guidance-vs-consensus, estimate-revision *trend* (yfinance gives point-in-time, not revision history) — later.
- The clear-role multi-agent restructure, per-role retry, macro-in-report, red-flag screens, incremental ROIC — later phases.
