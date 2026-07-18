"""Agent 9: Dispatcher (deterministic code, not an LLM).

Renders -> sends via Gmail SMTP (if configured) or Resend -> on failure,
writes report.html locally and surfaces a link. Delivery status is logged
back into state so the UI can show the pipeline completing either way.
"""

from __future__ import annotations

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from ..config import get_settings
from ..state import ResearchState
from .common import event


def _send_via_gmail(recipient: str, subject: str, html: str, settings) -> str | None:
    """Send via Gmail's own SMTP server, authenticated as the sender account.

    Returns None on success, or an error string on failure. Requires
    GMAIL_SENDER_EMAIL + a Gmail *App Password* (not the account password) -
    see .env.example.
    """
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = settings.gmail_sender_email
    msg["To"] = recipient
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as server:
            server.login(settings.gmail_sender_email, settings.gmail_app_password)
            server.sendmail(settings.gmail_sender_email, [recipient], msg.as_string())
        return None
    except Exception as exc:  # noqa: BLE001 - caller decides fallback
        return str(exc)


async def dispatcher(state: ResearchState) -> dict:
    settings = get_settings()
    settings.ensure_dirs()
    recipient = state.get("recipient_email")
    html = state.get("report_html", "")
    subject = f"Deep Research Report: {state.get('topic', 'Untitled')}"

    local_path = (settings.reports_dir / "report.html").resolve()
    Path(local_path).write_text(html, encoding="utf-8")
    local_link = local_path.as_uri()

    status: dict = {"delivered": False, "channel": None, "local_link": local_link}

    if recipient and settings.gmail_sender_email and settings.gmail_app_password:
        error = _send_via_gmail(recipient, subject, html, settings)
        if error is None:
            status.update(delivered=True, channel="gmail_smtp")
            msg = f"emailed report to {recipient} via Gmail SMTP ({settings.gmail_sender_email})"
        else:
            status.update(delivered=False, channel="fallback_link", error=error)
            msg = f"Gmail SMTP send failed ({error}); report available at {local_link}"
    elif recipient and settings.resend_api_key:
        try:
            import resend

            resend.api_key = settings.resend_api_key
            resp = resend.Emails.send(
                {
                    "from": settings.resend_from,
                    "to": [recipient],
                    "subject": subject,
                    "html": html,
                }
            )
            status.update(
                delivered=True,
                channel="resend",
                message_id=(resp or {}).get("id"),
            )
            msg = f"emailed report to {recipient} via Resend"
        except Exception as exc:  # noqa: BLE001 - fall back to hosted link
            status.update(delivered=False, channel="fallback_link", error=str(exc))
            msg = f"email failed ({exc}); report available at {local_link}"
    else:
        reason = "no recipient" if not recipient else "no email sender configured"
        status.update(channel="fallback_link", reason=reason)
        msg = f"no email sent ({reason}); report available at {local_link}"

    return {
        "delivery_status": status,
        "status_log": [event("Dispatcher", msg, **status)],
    }
