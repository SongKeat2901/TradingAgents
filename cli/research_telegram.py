"""Send research results to Telegram via Bot API on CLI exit.

Uses stdlib only (urllib + json + email.mime) to avoid pulling another
HTTP dep into the fork's install graph. The bot token is read from the
environment variable TRADINGRESEARCH_BOT_TOKEN; the chat id is passed
as a CLI flag (or auto-discovered from ~/.openclaw/openclaw.json).

Two delivery shapes:
- post_message: short inline text (≤ 4096 chars per Telegram cap).
- post_document: multipart/form-data file upload, used to attach the
  per-run PDF. Cap is 50MB per the Telegram Bot API.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


TELEGRAM_MAX_MESSAGE_LEN = 4096
_TRUNCATED_SUFFIX = "\n\n…(truncated)"
_RATING_EMOJI = {
    "OVERWEIGHT": "🟢", "BUY": "🟢", "HOLD": "🟡",
    "UNDERWEIGHT": "🔴", "SELL": "⛔",
}


class TelegramSendError(RuntimeError):
    """Raised when posting to Telegram fails."""


def _truncate_for_telegram(text: str) -> str:
    if len(text) <= TELEGRAM_MAX_MESSAGE_LEN:
        return text
    keep = TELEGRAM_MAX_MESSAGE_LEN - len(_TRUNCATED_SUFFIX)
    return text[:keep] + _TRUNCATED_SUFFIX


def post_message(bot_token: str, chat_id: str, text: str) -> dict:
    """Post one Telegram message via the Bot API. Returns the API result dict.

    Truncates messages over Telegram's 4096-char per-message limit.
    Raises TelegramSendError on transport failure or non-ok response.
    """
    text = _truncate_for_telegram(text)
    body = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode("utf-8")
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    req = urllib.request.Request(url, data=body, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise TelegramSendError(f"Failed to reach Telegram: {e}") from e
    except json.JSONDecodeError as e:
        raise TelegramSendError(f"Telegram returned non-JSON: {e}") from e

    if not payload.get("ok"):
        raise TelegramSendError(f"Telegram API returned not-ok: {payload}")
    return payload


def post_document(
    bot_token: str,
    chat_id: str,
    document_path: Path,
    caption: str = "",
    mime_type: str = "application/pdf",
    parse_mode: str | None = None,
) -> dict:
    """Upload a file to Telegram via sendDocument (multipart/form-data).

    Telegram's per-document cap is 50 MB. The caption (if provided) is
    truncated to 1024 chars per Telegram's per-caption cap.

    Stdlib-only: builds multipart body by hand (no `requests`).
    """
    boundary = f"----TgFormBoundary{uuid.uuid4().hex}"
    file_bytes = document_path.read_bytes()
    filename = document_path.name

    parts: list[bytes] = []

    def field(name: str, value: str) -> None:
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        )
        parts.append(value.encode("utf-8"))
        parts.append(b"\r\n")

    field("chat_id", chat_id)
    if caption:
        field("caption", caption[:1024])
    if parse_mode:
        field("parse_mode", parse_mode)

    parts.append(f"--{boundary}\r\n".encode())
    parts.append(
        (
            f'Content-Disposition: form-data; name="document"; '
            f'filename="{filename}"\r\n'
        ).encode()
    )
    parts.append(f"Content-Type: {mime_type}\r\n\r\n".encode())
    parts.append(file_bytes)
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())

    body = b"".join(parts)

    url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )

    try:
        # Larger timeout — we're uploading a multi-MB PDF
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise TelegramSendError(f"Failed to reach Telegram: {e}") from e
    except json.JSONDecodeError as e:
        raise TelegramSendError(f"Telegram returned non-JSON: {e}") from e

    if not payload.get("ok"):
        raise TelegramSendError(f"Telegram sendDocument not-ok: {payload}")
    return payload


def _md_clean(s: str) -> str:
    """Strip legacy-Markdown control chars from inline text (e.g. company names)
    so they can't break the caption's bold/italic parsing."""
    return re.sub(r"[*_`\[\]]", "", s or "").strip()


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _summary_fields(out: Path, ticker: str) -> dict:
    """Pull the at-a-glance fields for the Telegram caption from a run dir.
    Every field is best-effort — missing ones just drop out of the caption."""
    fields: dict = {"company": "", "rating": None, "ref": None, "ev_abs": None,
                    "fv": None, "mos": None, "next_earnings": None}
    fin = _read_json(out / "raw" / "financials.json") or {}
    m = re.search(r"^Name:\s*(.+)$", fin.get("fundamentals", ""), re.MULTILINE)
    if m:
        fields["company"] = _md_clean(m.group(1))
    try:
        from cli.daily_followup import parse_research
        parsed = parse_research(out)
    except Exception:  # noqa: BLE001 - never let the notifier crash the run
        parsed = None
    if parsed:
        fields["rating"] = (parsed.get("rating") or "").upper() or None
        fields["ref"] = parsed.get("reference_price")
        fields["ev_abs"] = parsed.get("ev")
    iv = _read_json(out / "raw" / "intrinsic_value.json")
    if iv:
        fields["fv"] = (iv.get("fair_value") or {}).get("base")
        fields["mos"] = iv.get("margin_of_safety_pct")
    cal = _read_json(out / "raw" / "calendar.json") or {}
    fields["next_earnings"] = (cal.get(ticker) or {}).get("next_expected")
    return fields


def _build_caption(out: Path, ticker: str, date: str) -> str:
    """A polished at-a-glance summary for the Telegram group (legacy Markdown)."""
    f = _summary_fields(out, ticker)
    rating = f["rating"]
    title = f"{f['company']} ({ticker})" if f["company"] else ticker
    emoji = _RATING_EMOJI.get(rating, "•")
    rating_disp = rating.title() if rating and rating != "UNKNOWN" else "—"
    lines = [
        f"📊 *TrueKnot Research* — {title}",
        f"{emoji} *{rating_disp}*  ·  as of {date} close",
        "",
    ]
    if f["ref"] is not None:
        lines.append(f"• Reference price:  ${f['ref']:,.2f}")
    if f["ev_abs"] is not None and f["ref"]:
        ev_pct = (f["ev_abs"] - f["ref"]) / f["ref"]
        lines.append(f"• 12-mo expected value:  *{ev_pct:+.1%}*  →  target ${f['ev_abs']:,.2f}")
    if f["fv"] is not None:
        mos = f"  (margin of safety {f['mos']:+.1%})" if f["mos"] is not None else ""
        lines.append(f"• Intrinsic fair value:  ${f['fv']:,.2f}{mos}")
    if f["next_earnings"]:
        lines.append(f"• Next catalyst:  ~{f['next_earnings']}")
    lines += [
        "",
        "📎 Full multi-agent research report attached (PDF).",
        "_TrueKnot Pte. Ltd. · trueknot.sg_",
    ]
    return "\n".join(lines)


def notify_success(
    bot_token: str,
    chat_id: str,
    output_dir: str,
    decision: str,
    *,
    ticker: str | None = None,
    date: str | None = None,
) -> None:
    """After a successful run, build a styled PDF and attach it to chat.

    Falls back to inline truncated text if PDF generation fails (so we
    never silently drop the result). The PDF route is preferred because
    Telegram's inline message cap is 4096 chars, which truncates any
    decision.md longer than ~half a page.

    `ticker` and `date` are optional — when None, they are read from the
    state.json in output_dir (kept for backward compat with existing
    callers that don't pass them).
    """
    out = Path(output_dir)

    if ticker is None or date is None:
        try:
            state = json.loads((out / "state.json").read_text(encoding="utf-8"))
            ticker = ticker or state.get("company_of_interest", "?")
            date = date or state.get("trade_date", "?")
        except (OSError, json.JSONDecodeError, KeyError):
            ticker = ticker or "?"
            date = date or "?"

    pdf_path: Path | None = None
    try:
        from cli.research_pdf import build_research_pdf

        pdf_path = build_research_pdf(
            output_dir=str(out), ticker=ticker, date=date, decision=decision
        )
    except Exception as e:  # noqa: BLE001 - fall back rather than fail the run
        # PDF is best-effort. Send the structured summary inline + flag the PDF
        # issue (the operator still has the on-disk reports for full detail).
        summary = _build_caption(out, ticker, date).replace(
            "📎 Full multi-agent research report attached (PDF).",
            f"⚠️ PDF unavailable ({type(e).__name__}); see on-disk reports.",
        )
        post_message(bot_token, chat_id, re.sub(r"[*_`]", "", summary))
        return

    caption = _build_caption(out, ticker, date)
    try:
        post_document(bot_token, chat_id, pdf_path, caption=caption, parse_mode="Markdown")
    except TelegramSendError:
        # A stray char tripped Markdown parsing — resend as plain text so the
        # report is never dropped.
        post_document(bot_token, chat_id, pdf_path, caption=re.sub(r"[*_`]", "", caption))


def notify_failure(
    bot_token: str, chat_id: str, ticker: str, date: str, error_summary: str
) -> None:
    """After a failed run, post the failure to chat."""
    text = f"❌ Research failed: {ticker} {date}\n\n{error_summary}"
    post_message(bot_token, chat_id, text)
