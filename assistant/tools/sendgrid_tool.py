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


def _build_attachments(attachments):
    """Read each file path into a SendGrid attachment dict. Returns (att_list, names) or
    raises ValueError with a readable message on a missing/unreadable file."""
    import base64
    import mimetypes
    import os
    if isinstance(attachments, str):
        paths = [p.strip() for p in attachments.split(",")]
    else:
        paths = [str(p).strip() for p in (attachments or [])]
    out, names = [], []
    for p in paths:
        if not p:
            continue
        if not os.path.isfile(p):
            raise ValueError(f"attachment not found: {p}")
        try:
            with open(p, "rb") as f:
                data = f.read()
        except OSError as e:
            raise ValueError(f"couldn't read attachment {p}: {e}")
        out.append({
            "content": base64.b64encode(data).decode(),
            "filename": os.path.basename(p),
            "type": mimetypes.guess_type(p)[0] or "application/octet-stream",
            "disposition": "attachment",
        })
        names.append(os.path.basename(p))
    return out, names


def send_email(to: str, subject: str, body: str, cc: str = None,
               html: bool = False, attachments: str = None) -> str:
    """Send an email via SendGrid from the configured sender. Confirms before sending.

    to/cc may be a single address or a comma-separated list. `attachments` is an optional
    comma-separated list of file paths to attach. Returns a readable status.
    """
    if not config.SENDGRID_ENABLED:
        return ("ERROR: SendGrid isn't configured — set SENDGRID_API_KEY and SENDGRID_FROM "
                "(a verified sender) in the .env file.")
    import requests
    from . import google_auth

    # Let recipients be given as a connected-account label ("main gmail") — resolve to email.
    to_list = google_auth.resolve_recipients(to)
    cc_list = google_auth.resolve_recipients(cc) if cc else []
    if not to_list:
        return "ERROR: no recipient address given."
    if not (subject or "").strip():
        return "ERROR: refusing to send an email with no subject."
    try:
        att_list, att_names = _build_attachments(attachments)
    except ValueError as e:
        return f"ERROR: {e}"

    who = ", ".join(to_list) + (f" (cc {', '.join(cc_list)})" if cc_list else "")
    extra = f" with {len(att_names)} attachment(s): {', '.join(att_names)}" if att_names else ""
    if not (not config.SENDGRID_CONFIRM_SENDS
            or approval.confirm_action(
                f"Send an email via SendGrid from {config.SENDGRID_FROM} to {who} — "
                f"subject '{subject}'{extra}?", always_ask=True)):
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
    if att_list:
        payload["attachments"] = att_list
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
        return (f"Sent to {who} from {config.SENDGRID_FROM}{extra}"
                + (f" (message id {mid})." if mid else "."))
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
        "description": "Send an email AS KARL via SendGrid from " + (config.SENDGRID_FROM or "the configured sender")
                       + ". This is the DEFAULT way to send email: whenever the user asks to send / "
                       "email someone WITHOUT specifying which of their own accounts to send from, "
                       "use this — Karl sends it as itself. Only if the user names a from-address / "
                       "their own account ('from my gmail', 'send as me') use send_message instead. "
                       "Confirms before sending.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address (or comma-separated addresses). A connected-account label like 'main gmail' is also accepted and resolves to that account's email."},
                "subject": {"type": "string", "description": "Email subject line."},
                "body": {"type": "string", "description": "Email body text."},
                "cc": {"type": "string", "description": "Optional cc address(es), comma-separated."},
                "html": {"type": "boolean", "description": "True to send the body as HTML (default plain text)."},
                "attachments": {"type": "string", "description": "Optional comma-separated file path(s) to attach (e.g. 'apartment_shopping_list.txt')."},
            },
            "required": ["to", "subject", "body"],
        },
    },
}
