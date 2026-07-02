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


def _collect_filings(submissions: dict, trade_date: datetime,
                     forms: tuple[str, ...], limit: int) -> list[dict]:
    """Most-recent-first list of up to `limit` filings of `forms` on/before
    trade_date (metadata only: form, date, accession, primary doc, 8-K items)."""
    recent = submissions.get("filings", {}).get("recent", {})
    out: list[dict] = []
    if not recent:
        return out
    items_arr = recent.get("items") or []
    for i in range(len(recent.get("form", []))):
        try:
            if recent["form"][i] not in forms:
                continue
            fd = datetime.strptime(recent["filingDate"][i], "%Y-%m-%d")
            if fd > trade_date:
                continue
            out.append({
                "form": recent["form"][i],
                "filing_date": recent["filingDate"][i],
                "accession_number": recent["accessionNumber"][i],
                "primary_document": recent["primaryDocument"][i],
                "items": items_arr[i] if i < len(items_arr) and items_arr[i] else None,
            })
            if len(out) >= limit:
                break
        except (ValueError, KeyError, IndexError):
            continue
    return out


def fetch_filing_surface(ticker: str, trade_date: str, max_8k: int = 5) -> dict[str, Any]:
    """Metadata-level SEC filing surface (FA-101 Phase 2b §9): the most recent
    8-K material events + the latest DEF 14A proxy filed on/before trade_date.
    Dates/item-codes/links only — no content parsing. Fail-soft ->
    {"unavailable": True, "reason": ...}."""
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

    def _url(f):
        return _FILING_URL.format(cik=cik, accession_no_dashes=f["accession_number"].replace("-", ""),
                                  primary_doc=f["primary_document"])
    eights = _collect_filings(submissions, td, ("8-K",), max_8k)
    proxies = _collect_filings(submissions, td, ("DEF 14A",), 1)
    for f in eights:
        f["url"] = _url(f)
    latest_proxy = proxies[0] if proxies else None
    if latest_proxy:
        latest_proxy["url"] = _url(latest_proxy)
    return {"ticker": ticker, "recent_8k": eights, "latest_def14a": latest_proxy, "source": "sec.gov"}


# 8-K item-code legend (the material-event categories most cited in research)
_8K_ITEMS = {
    "1.01": "material agreement", "1.03": "bankruptcy", "2.01": "acquisition/disposition",
    "2.02": "results of operations", "2.03": "new debt obligation", "3.01": "delisting notice",
    "4.01": "auditor change", "5.02": "officer/director change", "5.03": "bylaw change",
    "7.01": "Reg FD disclosure", "8.01": "other event", "9.01": "financial statements/exhibits",
}


def format_filing_surface_block(surface: dict[str, Any]) -> str:
    s = surface or {}
    if s.get("unavailable"):
        return (f"\n\n## SEC filing surface (8-K + proxy) — n/a "
                f"({s.get('reason', 'unavailable')})\n\n"
                "*No EDGAR filing metadata; do not cite 8-K/proxy dates.*\n")
    eights = s.get("recent_8k") or []
    proxy = s.get("latest_def14a")
    proxy_line = (f"filed {proxy['filing_date']} ([proxy]({proxy['url']}))"
                  if proxy else "none found on/before trade date")
    if not eights and not proxy:
        return ("\n\n## SEC filing surface (8-K + proxy) — none reported\n\n"
                "*No 8-K or DEF 14A on/before the trade date. Not a red flag.*\n")
    rows = ""
    for f in eights:
        codes = [c.strip() for c in (f.get("items") or "").split(",") if c.strip()]
        legend = ", ".join(f"{c} ({_8K_ITEMS.get(c, 'other')})" for c in codes) if codes else "—"
        rows += f"| {f['filing_date']} | {legend} |\n"
    block = (
        "\n\n## SEC filing surface (8-K events + latest proxy, EDGAR metadata)\n\n"
        f"**Latest DEF 14A (proxy — comp & governance):** {proxy_line}\n\n"
    )
    if eights:
        block += ("Recent 8-K material events:\n\n"
                  "| Date | Items |\n|---|---|\n" + rows)
    return block + (
        "\n*Dates/items/links only (not filing content). Cite verbatim; an 8-K "
        "clustering can flag a live catalyst. Item codes are SEC-standard.*\n"
    )


_EFTS_URL = "https://efts.sec.gov/LATEST/search-index?q={q}&forms={forms}"
_ACTIVIST_FORMS = "SC 13D,SC 13G,SC 13D/A,SC 13G/A"


def _parse_activist_hits(data: dict, cik10: str, td: datetime, company: str,
                         limit: int) -> list[dict]:
    """Pure: filter efts hits to 13D/13G naming this SUBJECT cik, on/before td,
    newest-first, deduped. Detection only — no thesis interpretation."""
    hits = (data or {}).get("hits", {}).get("hits", [])
    rows = []
    for h in hits:
        s = h.get("_source", {})
        if cik10 not in (s.get("ciks") or []):
            continue  # keep only filings where this company is a named party (a stake IN it)
        fd = s.get("file_date")
        try:
            if datetime.strptime(fd, "%Y-%m-%d") > td:
                continue
        except (ValueError, TypeError):
            continue
        forms = s.get("root_forms") or []
        form = forms[0] if forms else "SC 13"
        names = s.get("display_names") or []
        filers = [n for n in names if company.upper() not in (n or "").upper()] or names
        rows.append({"date": fd, "form": form, "activist": form.startswith("SC 13D"),
                     "filers": filers})
    rows.sort(key=lambda r: r["date"], reverse=True)
    out, seen = [], set()
    for r in rows:
        key = (r["date"], r["form"], tuple(r["filers"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
        if len(out) >= limit:
            break
    return out


def fetch_activist_filings(ticker: str, trade_date: str, limit: int = 6) -> dict[str, Any]:
    """Detect recent 13D (activist) / 13G (passive >5%) filings naming this
    company, via EDGAR full-text search filtered by the subject CIK (FA-101
    Phase 2b §8). Detection only — dates/filers/type, not the filing thesis.
    Fail-soft -> {"unavailable": True, "reason": ...}."""
    try:
        td = datetime.strptime(trade_date, "%Y-%m-%d")
    except ValueError:
        return {"unavailable": True, "reason": f"invalid trade_date: {trade_date}", "ticker": ticker}
    cik = _resolve_cik(ticker.upper())
    if cik is None:
        return {"unavailable": True, "reason": f"CIK not found for ticker {ticker}", "ticker": ticker}
    cik10 = f"{cik:010d}"
    sub_body = _http_get(_SUBMISSIONS_URL.format(cik=cik))
    company = ticker
    if sub_body is not None:
        try:
            company = json.loads(sub_body).get("name") or ticker
        except json.JSONDecodeError:
            pass
    import urllib.parse
    url = _EFTS_URL.format(q=urllib.parse.quote(f'"{company}"'),
                           forms=urllib.parse.quote(_ACTIVIST_FORMS))
    body = _http_get(url)
    if body is None:
        return {"unavailable": True, "reason": "EDGAR full-text search unreachable", "ticker": ticker}
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return {"unavailable": True, "reason": "efts JSON decode failed", "ticker": ticker}
    return {"ticker": ticker, "company": company,
            "filings": _parse_activist_hits(data, cik10, td, company, limit),
            "source": "sec.gov/efts"}


def format_activist_block(result: dict[str, Any]) -> str:
    r = result or {}
    if r.get("unavailable"):
        return (f"\n\n## Activist & large-stake filings (13D/13G) — n/a "
                f"({r.get('reason', 'unavailable')})\n\n"
                "*No EDGAR 13D/13G search result; do not cite activist stakes.*\n")
    filings = r.get("filings") or []
    if not filings:
        return ("\n\n## Activist & large-stake filings (13D/13G) — none reported\n\n"
                "*No 13D/13G filings naming this company on/before the trade date. "
                "Not a red flag (most names have none).*\n")
    rows = "".join(
        f"| {f['date']} | {f['form']} ({'activist' if f['activist'] else 'passive'}) | "
        f"{'; '.join(f['filers']) or 'n/a'} |\n"
        for f in filings
    )
    return (
        "\n\n## Activist & large-stake filings (13D/13G, EDGAR full-text)\n\n"
        "| Date | Type | Filer(s) |\n|---|---|---|\n" + rows +
        "\n*13D = >5% holder with activist intent; 13G = passive >5% holder. "
        "Dates/filers/type from EDGAR only — do NOT infer the filing's thesis or "
        "the current stake size (holdings change between amendments).*\n"
    )


# --- 8-K earnings press release (Exhibit 99.x) ---------------------------
# EARNINGS_RELEASE_GOAL.md (2026-07-02): the pro-deck gap analysis marked the
# capex-funding split and forward guidance "call-only", but both are in the
# free 8-K earnings press release (item 2.02 results, Ex-99.1) — verified on
# ORCL's 2026-06-10 8-K (Q1-FY27 guidance, the $40B debt+equity financing
# plan, RPO $455B→$638B, CEO/CFO quotes).

_INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/index.json"

# Excerpt targets for a press release: guidance language, backlog, the
# capex/financing funding structure, and the exec quote paragraphs (the quote
# text PRECEDES the "..., said <name>, Chief Executive Officer" attribution,
# hence the larger `before` window used in fetch_earnings_release).
# Order matters: excerpts are selected keyword-by-keyword against a shared
# char budget, so the highest-value targets come first and the generic
# "expect" catch-all last (MSFT live smoke: with "expect" second, its 3
# windows exhausted the budget before the exec-quote keywords ran).
RELEASE_EXCERPT_KEYWORDS = (
    "guidance",
    "chief executive officer",
    "chief financial officer",
    "remaining performance obligation",
    "financing",
    "capital expenditure",
    "expect",  # expects / expected to grow / expectations
)

_EX99_RE = re.compile(r"ex[-_]?99|99[._-]?1", re.IGNORECASE)
_EX99_FIRST_RE = re.compile(r"99[._d-]?1", re.IGNORECASE)


def _pick_ex99_document(names: list[str]) -> str | None:
    """Pick the press-release exhibit (Ex-99.x) from a filing-index file list.
    Prefers .htm/.html over .txt, and a 99.1-numbered exhibit over 99.2+.
    Returns None when the filing carries no Ex-99 document."""
    def _is_ex99(n: str) -> bool:
        return bool(_EX99_RE.search(n))

    cands = [n for n in names if _is_ex99(n) and n.lower().endswith((".htm", ".html"))]
    if not cands:
        cands = [n for n in names if _is_ex99(n) and n.lower().endswith(".txt")]
    if not cands:
        return None
    # 99.1 before 99.2+, then stable by name for determinism.
    cands.sort(key=lambda n: (0 if _EX99_FIRST_RE.search(n) else 1, n.lower()))
    return cands[0]


def fetch_earnings_release(
    ticker: str, trade_date: str, max_text_chars: int = 18_000
) -> dict[str, Any]:
    """Fetch the latest 8-K earnings press release (item 2.02, Exhibit 99.x)
    filed on or before trade_date.

    Returns {ticker, form, filing_date, accession_number, exhibit, url, items,
    content, content_truncated, excerpts, source} or fail-soft
    {"unavailable": True, "reason": ..., "ticker": ...}. Excerpts are always
    computed from the FULL stripped text (they feed the pm_brief block), using
    the same targeted-keyword approach as the 10-K truncation fix."""
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

    eights = _collect_filings(submissions, td, ("8-K",), limit=12)
    filing = next((f for f in eights if "2.02" in (f.get("items") or "")), None)
    if filing is None:
        return {"unavailable": True,
                "reason": "no item-2.02 (results) 8-K on/before trade_date",
                "ticker": ticker}

    accession_no_dashes = filing["accession_number"].replace("-", "")
    index_body = _http_get(_INDEX_URL.format(cik=cik, accession_no_dashes=accession_no_dashes))
    if index_body is None:
        return {"unavailable": True, "reason": "8-K filing index unreachable",
                "ticker": ticker, "filing_date": filing["filing_date"]}
    try:
        index = json.loads(index_body)
    except json.JSONDecodeError:
        return {"unavailable": True, "reason": "8-K filing index JSON decode failed",
                "ticker": ticker, "filing_date": filing["filing_date"]}
    names = [i.get("name", "") for i in index.get("directory", {}).get("item", [])]
    exhibit = _pick_ex99_document(names)
    if exhibit is None:
        return {"unavailable": True, "reason": "no Ex-99 press-release exhibit in the 8-K index",
                "ticker": ticker, "filing_date": filing["filing_date"]}

    url = _FILING_URL.format(cik=cik, accession_no_dashes=accession_no_dashes, primary_doc=exhibit)
    html = _http_get(url)
    if html is None:
        return {"unavailable": True, "reason": "press-release exhibit unreachable",
                "ticker": ticker, "filing_date": filing["filing_date"]}
    stripper = _HTMLStripper()
    try:
        stripper.feed(html.decode("utf-8", errors="replace"))
    except Exception:  # noqa: BLE001 — malformed exhibit HTML must fail soft
        return {"unavailable": True, "reason": "press-release HTML parse failed",
                "ticker": ticker, "filing_date": filing["filing_date"]}
    text = stripper.text

    excerpts: list[dict[str, Any]] = []
    try:
        excerpts = extract_keyword_excerpts(
            text, RELEASE_EXCERPT_KEYWORDS,
            before=800, after=1400, max_per_keyword=3, min_gap=1500,
            max_total_chars=14_000,
        )
    except Exception:  # noqa: BLE001 — excerpts must never block the fetch
        excerpts = []

    truncated = len(text) > max_text_chars
    if truncated:
        text = text[:max_text_chars]

    return {
        "ticker": ticker,
        "form": filing["form"],
        "filing_date": filing["filing_date"],
        "accession_number": filing["accession_number"],
        "exhibit": exhibit,
        "url": url,
        "items": filing.get("items"),
        "content": text,
        "content_truncated": truncated,
        "excerpts": excerpts,
        "source": "sec.gov",
    }


_RELEASE_HEADER = "## Latest earnings release (SEC 8-K Ex-99.1)"


def format_earnings_release_block(release: dict[str, Any] | None,
                                  max_head_chars: int = 1_500) -> str:
    """pm_brief.md block: filing metadata + head snippet + targeted excerpts.
    Honest n/a when unavailable — with an explicit do-not-fabricate directive."""
    r = release or {"unavailable": True, "reason": "not fetched"}
    if r.get("unavailable"):
        return (
            f"\n\n{_RELEASE_HEADER} — n/a ({r.get('reason', 'unavailable')})\n\n"
            "*No earnings press release available. Do not cite forward guidance, "
            "capex-funding plans, or management quotes from an earnings release — "
            "write 'not disclosed' instead of inventing them.*\n"
        )
    head = (r.get("content") or "").strip()[:max_head_chars]
    parts = [
        f"\n\n{_RELEASE_HEADER}\n\n"
        f"**Filed:** {r['filing_date']} (8-K item {r.get('items') or '2.02'})  \n"
        f"**Exhibit:** {r['exhibit']}  \n"
        f"**URL:** {r['url']}\n\n"
        "Verbatim press-release prose (public as of the trade date). Opening:\n\n"
        f"> {head}…\n"
    ]
    if r.get("excerpts"):
        parts.append("\nKey excerpts (full-text keyword search):\n")
        for e in r["excerpts"]:
            parts.append(f"\n**[{e['keyword']}]**\n\n> …{e['text'].strip()}…\n")
    parts.append(
        "\n*Quote forward guidance, the capex/financing funding structure, and "
        "management quotes from this release verbatim; anything the release does "
        "not state is 'not disclosed' — never paraphrase numbers or fill gaps "
        "from memory.*\n"
    )
    return "".join(parts)


def format_earnings_release_md(release: dict[str, Any]) -> str:
    """Render the full release dict as raw/earnings_release.md (same contract
    as format_for_prompt for the 10-Q/10-K). Returns "" when unavailable."""
    if not release or release.get("unavailable"):
        return ""
    truncation_note = (
        "\n\n*Content truncated for prompt budget; see exhibit URL for the full release.*"
        if release.get("content_truncated") else ""
    )
    excerpts_section = ""
    if release.get("excerpts"):
        parts = ["\n\n---\n\n### Targeted excerpts (full-document keyword search)\n"]
        for e in release["excerpts"]:
            parts.append(f"\n**[{e['keyword']}]**\n\n> …{e['text'].strip()}…\n")
        excerpts_section = "".join(parts)
    return (
        f"# Earnings press release — {release['ticker']} {release['form']} "
        f"filed {release['filing_date']} (Exhibit {release['exhibit']})\n\n"
        f"**Accession:** {release['accession_number']}  \n"
        f"**URL:** {release['url']}\n\n"
        "This is the verbatim text of the company's latest earnings press "
        "release (SEC 8-K item 2.02, Exhibit 99.x), public as of the trade "
        "date. It is the authoritative source for forward guidance, the "
        "capex/financing funding structure, RPO/backlog highlights, and "
        "management (CEO/CFO) quotes — quote it verbatim; never treat its "
        "contents as 'awaiting the call'.\n\n"
        "---\n\n"
        f"{release['content']}"
        f"{truncation_note}"
        f"{excerpts_section}\n"
    )


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

    html_text = html.decode("utf-8", errors="replace")

    # Phase 7.11: extract convertible-note specifics from inline-XBRL
    # BEFORE the HTML stripper destroys the `<ix:nonFraction>` tags. The
    # extractor is regex-based + tolerant; never raises. For 10-Qs without
    # convertibles, returns an empty list.
    convertibles: list[dict] = []
    try:
        from tradingagents.agents.utils.xbrl_convertibles import (
            extract_convertibles_from_html,
        )
        convertibles = extract_convertibles_from_html(html_text)
    except Exception:  # noqa: BLE001 — extraction must never block the fetch
        convertibles = []

    stripper = _HTMLStripper()
    try:
        stripper.feed(html_text)
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

    # Targeted excerpts come from the FULL text BEFORE truncation — this is
    # the whole point (the truncated head of a big inline-XBRL 10-K is
    # header metadata; the MD&A/RPO prose lives past the cut). Only attached
    # when truncation actually dropped content. Fail-open.
    excerpts: list[dict[str, Any]] = []
    if truncated:
        try:
            excerpts = extract_keyword_excerpts(text)
        except Exception:  # noqa: BLE001 — excerpts must never block the fetch
            excerpts = []

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
        "excerpts": excerpts,
        "convertibles": convertibles,
        "source": "sec.gov",
    }


# --- Targeted keyword excerpts (pro-deck technique B, 2026-07-02) ---
# The ORCL benchmark caught the 60K-char truncation silently dropping ALL
# MD&A prose on large inline-XBRL 10-Ks (header metadata alone exceeds 60K),
# so RPO / capex-guidance / prepayment / segment paragraphs never reached
# sec_filing.md. Fix: search the FULL stripped text for these topics and
# carry windowed excerpts past the truncation.
DEFAULT_EXCERPT_KEYWORDS = (
    "remaining performance obligation",
    "customer prepayment",
    "capital expenditure",
    "reportable segment",
    "operating segment",
)


def extract_keyword_excerpts(
    text: str,
    keywords: tuple[str, ...] = DEFAULT_EXCERPT_KEYWORDS,
    *,
    before: int = 300,
    after: int = 1700,
    max_per_keyword: int = 4,
    min_gap: int = 2000,
    max_total_chars: int = 32_000,
) -> list[dict[str, Any]]:
    """Windowed excerpts around keyword matches in the full filing text.

    Case-insensitive. Per keyword, keeps at most `max_per_keyword` matches
    that are at least `min_gap` chars apart; a match whose window overlaps an
    already-selected window (any keyword) is skipped (dedupe). Stops once the
    total excerpt budget `max_total_chars` is reached. Pure function.
    """
    if not text or not keywords:
        return []
    low = text.lower()
    selected: list[dict[str, Any]] = []
    windows: list[tuple[int, int]] = []
    total = 0
    for kw in keywords:
        kw_low = kw.lower()
        count = 0
        last_pos: int | None = None
        start = 0
        while count < max_per_keyword:
            i = low.find(kw_low, start)
            if i == -1:
                break
            start = i + len(kw_low)
            if last_pos is not None and i - last_pos < min_gap:
                continue
            w0, w1 = max(0, i - before), min(len(text), i + len(kw_low) + after)
            if any(w0 < e and s < w1 for s, e in windows):
                last_pos = i
                continue
            if total + (w1 - w0) > max_total_chars:
                return selected
            windows.append((w0, w1))
            selected.append({"keyword": kw, "position": i, "text": text[w0:w1]})
            total += w1 - w0
            last_pos = i
            count += 1
    return selected


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
    excerpts_section = ""
    if filing.get("excerpts"):
        parts = [
            "\n\n---\n\n### Targeted excerpts (full-document keyword search)\n\n"
            "The content above was truncated for prompt budget; these excerpts "
            "were located by searching the COMPLETE filing text for key topics. "
            "They are verbatim filing prose — quote them directly.\n"
        ]
        for e in filing["excerpts"]:
            parts.append(f"\n**[{e['keyword']}]**\n\n> …{e['text'].strip()}…\n")
        excerpts_section = "".join(parts)
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
        f"{truncation_note}"
        f"{excerpts_section}\n"
    )
