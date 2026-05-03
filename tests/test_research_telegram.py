"""Tests for cli.research_telegram (Telegram Bot API notifier)."""

from __future__ import annotations

import io
import json
from typing import Any
from urllib.error import URLError

import pytest

pytestmark = pytest.mark.unit


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _ok_response(extra: dict | None = None) -> _FakeResponse:
    payload: dict[str, Any] = {"ok": True, "result": {"message_id": 42}}
    if extra:
        payload.update(extra)
    return _FakeResponse(json.dumps(payload).encode("utf-8"))


def test_post_message_sends_correct_request(monkeypatch):
    from cli import research_telegram as t

    captured: dict[str, Any] = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["data"] = req.data
        captured["timeout"] = timeout
        return _ok_response()

    monkeypatch.setattr(t.urllib.request, "urlopen", fake_urlopen)

    result = t.post_message("BOT123", "-100", "hello")

    assert captured["url"] == "https://api.telegram.org/botBOT123/sendMessage"
    assert b"chat_id=-100" in captured["data"]
    assert b"text=hello" in captured["data"]
    assert captured["timeout"] == 15
    assert result["ok"] is True


def test_post_message_truncates_over_4096(monkeypatch):
    from cli import research_telegram as t

    captured: dict[str, Any] = {}

    def fake_urlopen(req, timeout=None):
        captured["data"] = req.data
        return _ok_response()

    monkeypatch.setattr(t.urllib.request, "urlopen", fake_urlopen)

    long_text = "x" * 5000
    t.post_message("BOT", "-100", long_text)

    sent = captured["data"].decode("utf-8")
    assert "…(truncated)" in sent or "%E2%80%A6%28truncated%29" in sent
    # Decoded length of the text= portion must not exceed 4096
    import urllib.parse as up

    sent_text = up.parse_qs(sent)["text"][0]
    assert len(sent_text) <= 4096


def test_post_message_raises_on_transport_failure(monkeypatch):
    from cli import research_telegram as t

    def fake_urlopen(req, timeout=None):
        raise URLError("connection refused")

    monkeypatch.setattr(t.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(t.TelegramSendError, match="Failed to reach Telegram"):
        t.post_message("BOT", "-100", "x")


def test_post_message_raises_on_not_ok(monkeypatch):
    from cli import research_telegram as t

    def fake_urlopen(req, timeout=None):
        return _FakeResponse(json.dumps({"ok": False, "description": "bad"}).encode())

    monkeypatch.setattr(t.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(t.TelegramSendError, match="not-ok"):
        t.post_message("BOT", "-100", "x")


def test_notify_success_builds_pdf_and_uploads(monkeypatch, tmp_path):
    """Happy path: notify_success builds PDF and posts via sendDocument multipart."""
    from cli import research_telegram as t

    # Stub the PDF builder so we don't need WeasyPrint in this test.
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake content")

    monkeypatch.setattr(
        "cli.research_pdf.build_research_pdf",
        lambda output_dir, ticker, date, decision: fake_pdf,
    )

    # state.json so notify_success can auto-discover ticker+date
    import json as _json
    (tmp_path / "state.json").write_text(_json.dumps({
        "company_of_interest": "NVDA",
        "trade_date": "2024-05-10",
    }), encoding="utf-8")

    captured: dict[str, Any] = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["data"] = req.data
        captured["headers"] = dict(req.headers)
        return _ok_response()

    monkeypatch.setattr(t.urllib.request, "urlopen", fake_urlopen)

    t.notify_success("BOT", "-100", str(tmp_path), "BUY")

    # Hits sendDocument, not sendMessage
    assert captured["url"].endswith("/sendDocument")
    # Body is multipart/form-data and contains the PDF bytes + caption
    content_type = captured["headers"].get("Content-type", "") or captured["headers"].get(
        "content-type", ""
    )
    assert "multipart/form-data" in content_type
    body = captured["data"]
    assert b"%PDF-1.4 fake content" in body
    assert b"NVDA" in body
    assert b"2024-05-10" in body
    assert b'name="document"' in body
    assert b'filename="fake.pdf"' in body


def test_notify_success_falls_back_to_inline_on_pdf_failure(monkeypatch, tmp_path):
    """If PDF generation raises, fall back to inline text via sendMessage."""
    from cli import research_telegram as t

    (tmp_path / "decision.md").write_text(
        "# NVDA — 2024-05-10\n\n**Decision:** BUY\n\nReasoning here.",
        encoding="utf-8",
    )

    def boom(**kwargs):
        raise RuntimeError("weasyprint not installed")

    monkeypatch.setattr("cli.research_pdf.build_research_pdf", boom)

    captured: dict[str, Any] = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["data"] = req.data
        return _ok_response()

    monkeypatch.setattr(t.urllib.request, "urlopen", fake_urlopen)

    t.notify_success("BOT", "-100", str(tmp_path), "BUY", ticker="NVDA", date="2024-05-10")

    # Falls back to sendMessage; body contains the warning + decision.md text
    assert captured["url"].endswith("/sendMessage")
    sent = captured["data"].decode("utf-8")
    assert "PDF+generation+failed" in sent or "PDF generation failed" in sent.replace("+", " ")
    assert "Reasoning here" in sent.replace("+", " ")


def test_notify_failure_includes_ticker_date_and_error(monkeypatch):
    from cli import research_telegram as t

    captured: dict[str, Any] = {}

    def fake_urlopen(req, timeout=None):
        captured["data"] = req.data
        return _ok_response()

    monkeypatch.setattr(t.urllib.request, "urlopen", fake_urlopen)

    t.notify_failure("BOT", "-100", "NVDA", "2024-05-10", "auth error: token expired")

    sent = captured["data"].decode("utf-8")
    # url-encoded; sanity check via decoding
    import urllib.parse as up

    text = up.parse_qs(sent)["text"][0]
    assert "Research failed" in text
    assert "NVDA" in text
    assert "2024-05-10" in text
    assert "auth error: token expired" in text
