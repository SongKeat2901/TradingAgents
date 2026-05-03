"""Send research results to Telegram via Bot API on CLI exit.

Uses stdlib only (urllib + json) to avoid pulling another dep into the
fork's install graph. The bot token is read from the environment variable
TRADINGRESEARCH_BOT_TOKEN; the chat id is passed as a CLI flag.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
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


def notify_success(
    bot_token: str, chat_id: str, output_dir: str, decision: str
) -> None:
    """After a successful run, post decision.md content to chat."""
    decision_md = (Path(output_dir) / "decision.md").read_text(encoding="utf-8")
    text = (
        f"✅ Decision: {decision}\n\n"
        f"{decision_md}\n\n"
        f"📁 Full reports: {output_dir}/"
    )
    post_message(bot_token, chat_id, text)


def notify_failure(
    bot_token: str, chat_id: str, ticker: str, date: str, error_summary: str
) -> None:
    """After a failed run, post the failure to chat."""
    text = f"❌ Research failed: {ticker} {date}\n\n{error_summary}"
    post_message(bot_token, chat_id, text)
