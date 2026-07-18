"""Dispatcher email delivery tests (offline - smtplib is mocked)."""

from __future__ import annotations

import pytest

import deep_researcher.agents.dispatcher as D
from deep_researcher.config import get_settings


@pytest.fixture(autouse=True)
def _isolated_reports_dir(monkeypatch, tmp_path):
    settings = get_settings()
    monkeypatch.setattr(settings, "reports_dir", tmp_path / "reports")
    monkeypatch.setattr(settings, "gmail_sender_email", None)
    monkeypatch.setattr(settings, "gmail_app_password", None)
    monkeypatch.setattr(settings, "resend_api_key", None)
    return settings


class _FakeSMTP:
    sent: list[tuple[str, list[str], str]] = []
    fail_login = False

    def __init__(self, host, port, timeout=None):
        self.host = host
        self.port = port

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def login(self, user, password):
        if _FakeSMTP.fail_login:
            raise Exception("bad app password")

    def sendmail(self, from_addr, to_addrs, msg):
        _FakeSMTP.sent.append((from_addr, to_addrs, msg))


@pytest.mark.asyncio
async def test_no_recipient_falls_back_to_local_link(_isolated_reports_dir):
    state = {"topic": "T", "report_html": "<p>hi</p>"}
    out = await D.dispatcher(state)
    status = out["delivery_status"]
    assert status["delivered"] is False
    assert status["channel"] == "fallback_link"
    assert status["reason"] == "no recipient"


@pytest.mark.asyncio
async def test_gmail_smtp_success(monkeypatch, _isolated_reports_dir):
    monkeypatch.setattr(_isolated_reports_dir, "gmail_sender_email", "a@gmail.com")
    monkeypatch.setattr(_isolated_reports_dir, "gmail_app_password", "app-pass")
    _FakeSMTP.sent = []
    _FakeSMTP.fail_login = False
    monkeypatch.setattr(D.smtplib, "SMTP_SSL", _FakeSMTP)

    state = {"topic": "T", "report_html": "<p>hi</p>", "recipient_email": "b@gmail.com"}
    out = await D.dispatcher(state)
    status = out["delivery_status"]

    assert status["delivered"] is True
    assert status["channel"] == "gmail_smtp"
    assert len(_FakeSMTP.sent) == 1
    from_addr, to_addrs, _msg = _FakeSMTP.sent[0]
    assert from_addr == "a@gmail.com"
    assert to_addrs == ["b@gmail.com"]


@pytest.mark.asyncio
async def test_gmail_smtp_failure_falls_back_to_local_link(monkeypatch, _isolated_reports_dir):
    monkeypatch.setattr(_isolated_reports_dir, "gmail_sender_email", "a@gmail.com")
    monkeypatch.setattr(_isolated_reports_dir, "gmail_app_password", "wrong-pass")
    _FakeSMTP.fail_login = True
    monkeypatch.setattr(D.smtplib, "SMTP_SSL", _FakeSMTP)

    state = {"topic": "T", "report_html": "<p>hi</p>", "recipient_email": "b@gmail.com"}
    out = await D.dispatcher(state)
    status = out["delivery_status"]

    assert status["delivered"] is False
    assert status["channel"] == "fallback_link"
    assert "bad app password" in status["error"]


@pytest.mark.asyncio
async def test_gmail_takes_priority_over_resend(monkeypatch, _isolated_reports_dir):
    monkeypatch.setattr(_isolated_reports_dir, "gmail_sender_email", "a@gmail.com")
    monkeypatch.setattr(_isolated_reports_dir, "gmail_app_password", "app-pass")
    monkeypatch.setattr(_isolated_reports_dir, "resend_api_key", "some-resend-key")
    _FakeSMTP.sent = []
    _FakeSMTP.fail_login = False
    monkeypatch.setattr(D.smtplib, "SMTP_SSL", _FakeSMTP)

    state = {"topic": "T", "report_html": "<p>hi</p>", "recipient_email": "b@gmail.com"}
    out = await D.dispatcher(state)
    assert out["delivery_status"]["channel"] == "gmail_smtp"
