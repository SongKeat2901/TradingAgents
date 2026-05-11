"""Deterministic SEC EDGAR filing fetcher (Phase-6.3 filing-anchor mitigation).

Run #2 of the Phase-6.2 validation (2026-05-05, MSFT trade date 2026-05-01)
caught the LLM hallucinating "the mid-May 10-Q is the binary catalyst pending
adjudication" — but MSFT files the 10-Q same-day as earnings (2026-04-29),
so the document was already public on the trade date. The agents missed
Azure +40% growth and Commercial RPO +99% YoY → $627B, both in the filed
10-Q, because the pipeline never fetched it.

This module pulls the most recent 10-Q or 10-K filed on or before the
trade date from SEC EDGAR (free, public, no auth) and extracts the primary
document text. PM Pre-flight writes the result to raw/sec_filing.md and
appends the filing date to the Reporting status table so downstream
agents see authoritative filing dates verbatim.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from datetime import datetime
from html.parser import HTMLParser
from typing import Any


_USER_AGENT = "TradingAgents Research-Verification songkeat@gmail.com"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_FILING_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{primary_doc}"


# Common tickers cached so we don't need to fetch the full directory on every
# run. yfinance does not expose CIKs; SEC's directory is the canonical source.
_TICKER_CIK: dict[str, int] = {
    "MSFT": 789019,
    "AAPL": 320193,
    "GOOGL": 1652044,
    "GOOG": 1652044,
    "AMZN": 1018724,
    "META": 1326801,
    "NVDA": 1045810,
    "TSLA": 1318605,
    "NFLX": 1065280,
    "ORCL": 1341439,
    "CRM": 1108524,
    "ADBE": 796343,
    "AMD": 2488,
    "INTC": 50863,
    "MU": 723125,
    "AVGO": 1730168,
    "MRVL": 1835632,
    "MARA": 1507605,
    "RIOT": 1167419,
    "COIN": 1679788,
    "CRWD": 1535527,
    "SNOW": 1640147,
    "PLTR": 1321655,
    "UBER": 1543151,
    "SHOP": 1594805,
    "BABA": 1577552,
    "JPM": 19617,
    "BAC": 70858,
    "BRK-B": 1067983,
    "V": 1403161,
    "MA": 1141391,
}


def _http_get(url: str, timeout: int = 30) -> bytes | None:
    """Fetch a URL with a proper User-Agent. SEC blocks default UA. Returns
    None on any error (network, HTTP status, timeout). Caller decides fallback."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return None


def _resolve_cik(ticker: str) -> int | None:
    """Map ticker → CIK, with fallback to the full SEC directory on cache miss."""
    if ticker in _TICKER_CIK:
        return _TICKER_CIK[ticker]
    body = _http_get(_TICKERS_URL)
    if body is None:
        return None
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None
    # Format: {"0": {"cik_str": 789019, "ticker": "MSFT", "title": "..."}, ...}
    for entry in data.values():
        if entry.get("ticker", "").upper() == ticker.upper():
            cik = int(entry["cik_str"])
            _TICKER_CIK[ticker.upper()] = cik
            return cik
    return None


class _HTMLStripper(HTMLParser):
    """Strip HTML tags while skipping script/style content."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style"):
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)

    @property
    def text(self) -> str:
        text = " ".join(self._parts)
        return re.sub(r"\s+", " ", text).strip()


def _find_recent_filing(
    submissions: dict, trade_date: datetime, forms: tuple[str, ...] = ("10-Q", "10-K")
) -> dict | None:
    """Find the most recent 10-Q or 10-K filed on or before trade_date.
    Returns a dict with form, filingDate, accessionNumber, primaryDocument."""
    recent = submissions.get("filings", {}).get("recent", {})
    if not recent:
        return None
    for i in range(len(recent.get("form", []))):
        try:
            if recent["form"][i] not in forms:
                continue
            filing_date = datetime.strptime(recent["filingDate"][i], "%Y-%m-%d")
            if filing_date > trade_date:
                continue
            return {
                "form": recent["form"][i],
                "filing_date": recent["filingDate"][i],
                "accession_number": recent["accessionNumber"][i],
                "primary_document": recent["primaryDocument"][i],
            }
        except (ValueError, KeyError, IndexError):
            continue
    return None


def fetch_latest_filing(
    ticker: str, trade_date: str, max_text_chars: int = 60_000
) -> dict[str, Any]:
    """Fetch the most recent 10-Q or 10-K filed on or before trade_date.

    Returns either:
        {
            "ticker": str,
            "form": "10-Q" | "10-K",
            "filing_date": "YYYY-MM-DD",
            "accession_number": "0001234567-26-123456",
            "primary_document": "msft-20260331.htm",
            "url": "https://www.sec.gov/Archives/edgar/data/...",
            "content": "<plain-text content, truncated>",
            "content_truncated": bool,
            "source": "sec.gov",
        }
        OR
        {"unavailable": True, "reason": "<short string>", "ticker": str}
    """
    try:
        td = datetime.strptime(trade_date, "%Y-%m-%d")
    except ValueError:
        return {"unavailable": True, "reason": f"invalid trade_date: {trade_date}", "ticker": ticker}

    cik = _resolve_cik(ticker.upper())
    if cik is None:
        return {"unavailable": True, "reason": f"CIK not found for ticker {ticker}", "ticker": ticker}

    body = _http_get(_SUBMISSIONS_URL.format(cik=cik))
    if body is None:
        return {"unavailable": True, "reason": "EDGAR submissions endpoint unreachable", "ticker": ticker}

    try:
        submissions = json.loads(body)
    except json.JSONDecodeError:
        return {"unavailable": True, "reason": "EDGAR submissions JSON decode failed", "ticker": ticker}

    filing = _find_recent_filing(submissions, td)
    if filing is None:
        return {"unavailable": True, "reason": "no 10-Q or 10-K filed on or before trade_date", "ticker": ticker}

    accession_no_dashes = filing["accession_number"].replace("-", "")
    url = _FILING_URL.format(
        cik=cik,
        accession_no_dashes=accession_no_dashes,
        primary_doc=filing["primary_document"],
    )

    html = _http_get(url)
    if html is None:
        return {
            "unavailable": True,
            "reason": "filing document unreachable",
            "ticker": ticker,
            "form": filing["form"],
            "filing_date": filing["filing_date"],
        }

    stripper = _HTMLStripper()
    try:
        stripper.feed(html.decode("utf-8", errors="replace"))
    except Exception:
        return {
            "unavailable": True,
            "reason": "filing HTML parse failed",
            "ticker": ticker,
            "form": filing["form"],
            "filing_date": filing["filing_date"],
        }
    text = stripper.text
    truncated = len(text) > max_text_chars
    if truncated:
        text = text[:max_text_chars]

    return {
        "ticker": ticker,
        "form": filing["form"],
        "filing_date": filing["filing_date"],
        "accession_number": filing["accession_number"],
        "primary_document": filing["primary_document"],
        "url": url,
        "content": text,
        "content_truncated": truncated,
        "source": "sec.gov",
    }


_XBRL_SIGNATURES = ("us-gaap:", "iso4217:", "xbrli:", "<ix:")


def _looks_like_xbrl(content: str) -> bool:
    """Detect inline-XBRL encoded 10-Q/10-K content.

    Real 10-Q text mentions GAAP concepts in prose; XBRL-encoded content
    embeds them as machine-readable tag tokens like `us-gaap:CommonStockMember`,
    `iso4217:USD`, `xbrli:shares`. We require ≥3 distinct XBRL tokens
    (single-mention may be a legitimate prose reference) before flagging.
    """
    if not content:
        return False
    hits = sum(1 for sig in _XBRL_SIGNATURES if sig in content)
    if hits >= 2:
        return True
    # Density check: many `us-gaap:` tokens specifically is a strong tell
    return content.count("us-gaap:") >= 3


_XBRL_WARNING = (
    "\n\n> ⚠️ **XBRL ENCODING WARNING**: This filing's content is "
    "inline-XBRL encoded. Numerical data and structural tagging are "
    "machine-readable but **prose footnotes (e.g. \"Note 5\", "
    "\"Note 15 — Subsequent Events\") are NOT available as readable "
    "text**.\n\n"
    "**Do NOT cite specific Note numbers** as if quoting a prose "
    "footnote (e.g. \"per Subsequent Event Note 15\" or \"Note 14 "
    "discloses $X\"). The Note structure is encoded in XBRL tags, "
    "not narrative prose, and any citation that pretends to read the "
    "Note's text is fabricated attribution.\n\n"
    "**Treat narrative claims** (acquisition specifics, segment "
    "commentary, dilution scenarios) **as news-sourced** unless you "
    "can verify the exact dollar/share figure in `raw/financials.json` "
    "(balance_sheet, income_statement, cashflow cells). When in doubt, "
    "attribute to news.json / social.json rather than fabricating a "
    "10-Q footnote cite.\n"
)


def format_for_prompt(filing: dict[str, Any]) -> str:
    """Render a filing dict as a Markdown block for raw/sec_filing.md.
    Returns "" if the filing is unavailable."""
    if filing.get("unavailable"):
        return ""
    truncation_note = (
        "\n\n*Content truncated for prompt budget; see filing URL for full document.*"
        if filing.get("content_truncated")
        else ""
    )
    xbrl_warning = _XBRL_WARNING if _looks_like_xbrl(filing.get("content", "")) else ""
    return (
        f"# SEC Filing — {filing['ticker']} {filing['form']} filed {filing['filing_date']}\n\n"
        f"**Accession:** {filing['accession_number']}  \n"
        f"**Document:** {filing['primary_document']}  \n"
        f"**URL:** {filing['url']}\n\n"
        "This is the verbatim text of the most recent 10-Q or 10-K filed on "
        "or before the trade date. **The numbers in this filing are public "
        "knowledge as of the trade date; treat them as known data, NEVER as "
        "'pending adjudication' or 'awaiting filing'.** Look for "
        "Remaining Performance Obligations (RPO), segment revenue and "
        "operating income, capital expenditures, and Azure / cloud growth "
        "rates in the MD&A section."
        f"{xbrl_warning}\n\n"
        "---\n\n"
        f"{filing['content']}"
        f"{truncation_note}\n"
    )
