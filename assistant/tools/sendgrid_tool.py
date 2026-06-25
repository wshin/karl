"""Outbound email via SendGrid.

Separate from gmail_tool.send_message (which sends through a connected Google account):
this sends through SendGrid's HTTP API from a fixed verified sender (config.SENDGRID_FROM).
Enabled only when SENDGRID_API_KEY + SENDGRID_FROM are set. Every send is confirmed first
(approval.confirm_action, always_ask) — outward action, no auto-send bypass.
"""
import logging

import approval
import config

log = logging.getLogger("assistant.sendgrid")

_ENDPOINT = "https://api.sendgrid.com/v3/mail/send"


def send_email(to: str, subject: str, body: str, cc: str = None,
               html: bool = False) -> str:
    """Send an email via SendGrid from the configured sender. Confirms before sending.

    to/cc may be a single address or a comma-separated list. Returns a readable status.
    """
    if not config.SENDGRID_ENABLED:
        return ("ERROR: SendGrid isn't configured — set SENDGRID_API_KEY and SENDGRID_FROM "
                "(a verified sender) in the .env file.")
    import requests

    to_list = [a.strip() for a in (to or "").split(",") if a.strip()]
    cc_list = [a.strip() for a in (cc or "").split(",") if a.strip()]
    if not to_list:
        return "ERROR: no recipient address given."
    if not (subject or "").strip():
        return "ERROR: refusing to send an email with no subject."

    who = ", ".join(to_list) + (f" (cc {', '.join(cc_list)})" if cc_list else "")
    if not (not config.SENDGRID_CONFIRM_SENDS
            or approval.confirm_action(
                f"Send an email via SendGrid from {config.SENDGRID_FROM} to {who} — "
                f"subject '{subject}'?", always_ask=True)):
        return "DENIED: email not sent (you declined)."

    personalization = {"to": [{"email": a} for a in to_list]}
    if cc_list:
        personalization["cc"] = [{"email": a} for a in cc_list]
    payload = {
        "personalizations": [personalization],
        "from": {"email": config.SENDGRID_FROM, "name": config.SENDGRID_FROM_NAME},
        "subject": subject,
        "content": [{"type": "text/html" if html else "text/plain", "value": body or ""}],
    }
    try:
        r = requests.post(
            _ENDPOINT,
            headers={"Authorization": f"Bearer {config.SENDGRID_API_KEY}",
                     "Content-Type": "application/json"},
            json=payload, timeout=30)
    except Exception as e:  # noqa: BLE001
        return f"ERROR: couldn't reach SendGrid ({e})."

    if r.status_code in (200, 202):
        mid = r.headers.get("X-Message-Id", "")
        return f"Sent to {who} from {config.SENDGRID_FROM}" + (f" (message id {mid})." if mid else ".")
    # Surface SendGrid's error (e.g. unverified sender → 403) so the user can fix it.
    detail = ""
    try:
        errs = r.json().get("errors", [])
        detail = "; ".join(e.get("message", "") for e in errs) or r.text[:300]
    except Exception:  # noqa: BLE001
        detail = r.text[:300]
    log.debug("sendgrid send failed %s: %s", r.status_code, detail)
    return f"ERROR: SendGrid rejected the send ({r.status_code}): {detail}"


SEND_EMAIL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "send_email",
        "description": "Send an email via SendGrid from " + (config.SENDGRID_FROM or "the configured sender")
                       + ". Use this when the user asks to send/email someone via SendGrid (distinct "
                       "from send_message, which sends from a connected Gmail account). Confirms first.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address (or comma-separated addresses)."},
                "subject": {"type": "string", "description": "Email subject line."},
                "body": {"type": "string", "description": "Email body text."},
                "cc": {"type": "string", "description": "Optional cc address(es), comma-separated."},
                "html": {"type": "boolean", "description": "True to send the body as HTML (default plain text)."},
            },
            "required": ["to", "subject", "body"],
        },
    },
}
