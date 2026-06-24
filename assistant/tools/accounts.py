"""Tools for the user to manage Karl's connected Google accounts from inside a chat:
list what's connected, and connect a new account (runs the OAuth browser flow).
"""
import logging
import os

import config

log = logging.getLogger("assistant.accounts")


def list_google_accounts() -> str:
    """List the Google (Gmail + Calendar) accounts Karl is connected to, by email and
    (if the user set one) their chosen label."""
    from . import google_auth
    connected = google_auth.available_accounts()
    lines = []
    for acct in connected:
        email = google_auth.account_email(acct) or "(address unavailable)"
        label = google_auth.account_label(acct)
        primary = " — primary" if acct == google_auth.primary_account() else ""
        # Lead with the email; show the user's label after it when one is set.
        lines.append(f"- {email}" + (f"  (label: {label})" if label else "") + primary)
    out = "Connected Google accounts:\n" + ("\n".join(lines) if lines else "  (none yet)")
    pending = [a for a in config.GOOGLE_ACCOUNTS if a not in connected]
    if pending:
        out += ("\nConfigured but not yet connected: " + ", ".join(pending)
                + " — ask me to connect one and I'll open the browser.")
    return out


def set_account_label(account: str, label: str) -> str:
    """Give a connected account a custom display label (or update its existing one), so
    Karl refers to it by that name instead of the email. `account` may be an email, the
    current label, or the internal key."""
    from . import google_auth
    if not (label or "").strip():
        return "Tell me what label to use (e.g. 'personal', 'work', 'main inbox')."
    email = google_auth.account_email(account)
    if not email and google_auth.resolve_account(account) not in google_auth.available_accounts():
        return (f"I don't have a connected account matching '{account}'. "
                "Ask me to list your accounts to see the options.")
    google_auth.set_account_label(account, label)
    return f"Done — I'll call {email or account} \"{label.strip()}\" from now on."


def clear_account_label(account: str) -> str:
    """Remove a connected account's custom label so Karl refers to it by its email again.
    `account` may be an email, the current label, or the internal key."""
    from . import google_auth
    had = google_auth.account_label(account)
    email = google_auth.account_email(account)
    if not had:
        return f"{email or account} doesn't have a custom label — I already use its email."
    google_auth.clear_account_label(account)
    return f"Removed the \"{had}\" label — I'll refer to {email or account} by its email now."


def connect_google_account(account: str = None) -> str:
    """Connect a Google account by running the OAuth flow (opens a browser). Call this
    only once the user is ready, since it opens a browser they must complete. `account`
    (a label) is OPTIONAL — omit it to just connect and identify the account by its email
    address; only pass it if the user explicitly wants a custom name for it."""
    from . import google_auth
    account = (account or "").strip().lower()
    if not os.path.exists(config.GOOGLE_CREDENTIALS_PATH):
        return (
            "First-time Google setup is needed. This credentials.json is created ONCE and "
            "then connects ANY account — personal OR business — so you don't repeat it per "
            "account:\n"
            "1. In the Google Cloud console, create a project.\n"
            "2. Enable the Gmail API and the Google Calendar API.\n"
            "3. OAuth consent screen → 'External' (use 'Internal' only if you want to "
            "restrict it to your own Workspace org) → publish to production.\n"
            "4. Create an OAuth client ID of type 'Desktop app' and download the JSON.\n"
            f"5. Save it as credentials.json in the Karl folder ({config.GOOGLE_CREDENTIALS_PATH}).\n"
            "Then ask me to connect the account. Note the account-type difference at sign-in:\n"
            "- Personal Gmail: connects directly (click through the one-time 'unverified app' "
            "screen).\n"
            "- Business/Workspace Gmail: your IT admin may block third-party apps — if it "
            "won't connect, ask them to allow it, or have a Workspace admin create the OAuth "
            "app as 'Internal'.")
    # No label given — connect and identify the account by its email (the common case).
    if not account:
        try:
            _key, email = google_auth.authorize_new()  # browser flow; derives a key from the email
        except Exception as e:  # noqa: BLE001
            return f"Couldn't connect the account: {e}"
        return (f"Connected {email or 'the account'} — it's now included in your email and "
                "calendar. I'll refer to it by its email; just say so if you'd like to give "
                "it a label.")
    # Explicit label requested.
    if account in google_auth.available_accounts():
        email = google_auth.account_email(account) or ""
        return f"'{account}'{f' ({email})' if email else ''} is already connected."
    try:
        google_auth.authorize(account)  # opens the browser; blocks until the user finishes
    except Exception as e:  # noqa: BLE001
        return f"Couldn't connect '{account}': {e}"
    email = google_auth.account_email(account) or ""
    return (f"Connected '{account}'{f' ({email})' if email else ''} — it's now included in "
            "your email and calendar.")


LIST_ACCOUNTS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_google_accounts",
        "description": "List which Google (Gmail + Calendar) accounts Karl is connected to, "
                       "with their email addresses. Use when the user asks what accounts "
                       "you're connected to.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

CONNECT_ACCOUNT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "connect_google_account",
        "description": "Connect a new Google account by opening the OAuth browser flow. The "
                       "user signs in with that account and approves. Call this ONLY after "
                       "confirming the user is ready (it opens a browser they must complete). "
                       "Do NOT ask the user for a label — just call it with no arguments and "
                       "the account is identified by its email address. Only pass `account` if "
                       "the user explicitly volunteers a custom name. If first-time setup is "
                       "missing it returns the steps to relay.",
        "parameters": {
            "type": "object",
            "properties": {"account": {"type": "string", "description": "OPTIONAL custom label. Omit unless the user explicitly asks to name the account; by default the account is identified by its email."}},
            "required": [],
        },
    },
}

SET_ACCOUNT_LABEL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "set_account_label",
        "description": "Give a connected Google account a custom display label / nickname (or "
                       "change its current one), so you refer to it by that name instead of its "
                       "email address. Use when the user says to call/label/name an account "
                       "something (e.g. 'call wontaek@gmail.com my personal account').",
        "parameters": {
            "type": "object",
            "properties": {
                "account": {"type": "string", "description": "Which account — its email address, current label, or internal key."},
                "label": {"type": "string", "description": "The label/nickname to use (e.g. 'personal', 'work', 'main inbox')."},
            },
            "required": ["account", "label"],
        },
    },
}

CLEAR_ACCOUNT_LABEL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "clear_account_label",
        "description": "Remove a connected Google account's custom label so you refer to it by "
                       "its email address again. Use when the user says to remove/clear/reset "
                       "an account's label or nickname.",
        "parameters": {
            "type": "object",
            "properties": {
                "account": {"type": "string", "description": "Which account — its email address, current label, or internal key."},
            },
            "required": ["account"],
        },
    },
}
