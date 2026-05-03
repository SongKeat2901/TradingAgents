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
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


TELEGRAM_MAX_MESSAGE_LEN = 4096
_TRUNCATED_SUFFIX = "\n\n…(truncated)"


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
        # PDF is best-effort. Fall back to inline truncated text.
        decision_md = (out / "decision.md").read_text(encoding="utf-8")
        text = (
            f"✅ {ticker} {date} — {decision}\n\n"
            f"⚠️ PDF generation failed ({type(e).__name__}: {e}); inline below.\n\n"
            f"{decision_md}\n\n"
            f"📁 Full reports: {out}/"
        )
        post_message(bot_token, chat_id, text)
        return

    caption = (
        f"✅ {ticker} {date}\n"
        f"Decision: {decision}\n\n"
        f"Full multi-agent research attached as PDF. "
        f"On-disk reports: {out}/"
    )
    post_document(bot_token, chat_id, pdf_path, caption=caption)


def notify_failure(
    bot_token: str, chat_id: str, ticker: str, date: str, error_summary: str
) -> None:
    """After a failed run, post the failure to chat."""
    text = f"❌ Research failed: {ticker} {date}\n\n{error_summary}"
    post_message(bot_token, chat_id, text)
