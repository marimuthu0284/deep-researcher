"""Agent 9: Dispatcher (deterministic code, not an LLM).

Renders -> sends via Resend -> on failure, writes report.html locally and
surfaces a link. Delivery status is logged back into state so the UI can show
the pipeline completing either way.
"""

from __future__ import annotations

from pathlib import Path

from ..config import get_settings
from ..state import ResearchState
from .common import event


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

    if recipient and settings.resend_api_key:
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
        reason = "no recipient" if not recipient else "no RESEND_API_KEY"
        status.update(channel="fallback_link", reason=reason)
        msg = f"no email sent ({reason}); report available at {local_link}"

    return {
        "delivery_status": status,
        "status_log": [event("Dispatcher", msg, **status)],
    }
