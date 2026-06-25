"""Shared Google OAuth for the Calendar and Gmail tools — multi-account.

Each connected Google account (config.GOOGLE_ACCOUNTS, e.g. work + personal) has its
own token file: the FIRST label keeps the original token.json (no re-auth), the rest
use token_<label>.json beside it. Authorize one at a time, signing in with that
account in the browser:

    python assistant/tools/google_auth.py <account>     # e.g. personal

Run with no argument to (re-)authorize the primary account. Adding a scope re-consents
automatically (the saved token's granted scopes are checked). The google-* libraries
are imported lazily so the rest of Karl runs without them.
"""
import logging
import os
import re
import sys

# Allow running directly for one-time OAuth setup.
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

log = logging.getLogger("assistant.google")

_services: dict = {}  # (api, version, account) -> client


def primary_account() -> str:
    return config.GOOGLE_ACCOUNTS[0] if config.GOOGLE_ACCOUNTS else "default"


def _token_path(account: str) -> str:
    """Token file for an account: the first label keeps token.json (back-compat)."""
    if account == primary_account():
        return config.GOOGLE_TOKEN_PATH
    return os.path.join(os.path.dirname(config.GOOGLE_TOKEN_PATH), f"token_{account}.json")


def available_accounts() -> list:
    """Accounts with a saved token — discovered from disk (token.json for the primary,
    token_<label>.json for the rest), so a newly-connected label appears even if it
    isn't in GOOGLE_ACCOUNTS. Primary first."""
    accts = []
    if os.path.exists(config.GOOGLE_TOKEN_PATH):
        accts.append(primary_account())
    try:
        for fn in sorted(os.listdir(os.path.dirname(config.GOOGLE_TOKEN_PATH))):
            m = re.match(r"token_(.+)\.json$", fn)
            # skip internal/temp keys (e.g. the "_pending" token used mid-connect)
            if m and m.group(1) != primary_account() and not m.group(1).startswith("_"):
                accts.append(m.group(1))
    except OSError:
        pass
    return accts


def authorize(account: str):
    """Run the OAuth flow for `account` (opens a browser if it isn't already connected)."""
    return _credentials(account, interactive=True)


def disconnect(account: "str | None") -> bool:
    """Remove a connected account: delete its saved token (so Karl stops accessing it),
    drop its display label, and clear cached clients/emails. `account` may be a key,
    email, or label. Returns True if a token was actually removed. credentials.json (the
    shared app) is never touched, so the account can be reconnected later."""
    key = resolve_account(account) or primary_account()
    removed = False
    try:
        os.unlink(_token_path(key))
        removed = True
    except OSError:
        pass
    _email_cache.pop(key, None)
    for k in [k for k in _services if k[2:] == (key,)]:   # cached service clients for it
        _services.pop(k, None)
    labels = load_account_labels()
    if labels.pop(key, None) is not None:
        _save_account_labels(labels)
    return removed


def _stable_key_from_email(email: str, taken: set) -> str:
    """A filename-safe internal key derived from an email (e.g. wontaek@gmail.com ->
    'wontaek'), made unique against the already-connected keys in `taken`."""
    email = (email or "").lower()
    base = re.sub(r"[^a-z0-9]+", "", email.partition("@")[0]) or "account"
    if base not in taken:
        return base
    dom = re.sub(r"[^a-z0-9]+", "", email.partition("@")[2].split(".")[0])
    cand = f"{base}{dom}" if dom and f"{base}{dom}" not in taken else base
    i = 2
    while cand in taken:
        cand, i = f"{base}{i}", i + 1
    return cand


def authorize_new(interactive: bool = True) -> tuple:
    """Connect a brand-new account WITHOUT a user-supplied label: run OAuth into a temp
    token, read the account's email, then settle the token under a stable key derived
    from that email. If the email is already connected, reuse it. Returns (key, email)."""
    existing = available_accounts()
    tmp = "_pending"
    tmp_path = _token_path(tmp)
    if os.path.exists(tmp_path):
        try:
            os.unlink(tmp_path)               # never reuse a stale pending token
        except OSError:
            pass
    try:
        _credentials(tmp, interactive=interactive)        # browser consent
        _email_cache.pop(tmp, None)                        # force a fresh lookup for this token
        email = account_email(tmp)
    except Exception:
        try:
            os.path.exists(tmp_path) and os.unlink(tmp_path)
        except OSError:
            pass
        _services.pop(("gmail", "v1", tmp), None)
        _email_cache.pop(tmp, None)
        raise
    # already connected under another key? reuse it, drop the temp token.
    for a in existing:
        em = account_email(a)
        if em and email and em.lower() == email.lower():
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            _services.pop(("gmail", "v1", tmp), None)
            return a, email
    key = _stable_key_from_email(email, {a.lower() for a in existing} | {primary_account()})
    os.replace(tmp_path, _token_path(key))
    _services.clear()                         # temp-keyed clients are now stale
    _email_cache.clear()
    return key, email


_email_cache: dict = {}  # label -> email address (network lookup is cached)


def account_email(account: str) -> "str | None":
    """The email address of a connected account (via Gmail getProfile), or None."""
    key = account or primary_account()
    if key in _email_cache:
        return _email_cache[key]
    try:
        email = service("gmail", "v1", account).users().getProfile(
            userId="me").execute().get("emailAddress")
    except Exception as e:  # noqa: BLE001
        log.debug("account_email(%s) failed: %s", account, e)
        email = None
    _email_cache[key] = email
    return email


def account_map() -> dict:
    """{label: email} for every connected account (emails cached after first lookup)."""
    return {a: account_email(a) for a in available_accounts()}


# --- user-chosen display labels (nicknames) for accounts ----------------------
# Optional aliases on top of the email, e.g. {"personal": "main"}. Keyed by the internal
# account key. When set, Karl refers to the account by the label; clearing reverts to the
# email. Stored in config.GOOGLE_LABELS_PATH.
def load_account_labels() -> dict:
    import json
    try:
        with open(config.GOOGLE_LABELS_PATH, "r", encoding="utf-8") as f:
            return {str(k): str(v) for k, v in json.load(f).items() if v}
    except (OSError, ValueError):
        return {}


def _save_account_labels(labels: dict) -> None:
    import json
    tmp = config.GOOGLE_LABELS_PATH + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(labels, f)
        os.replace(tmp, config.GOOGLE_LABELS_PATH)
    except OSError as e:
        log.debug("could not write account labels: %s", e)


def account_label(account: "str | None") -> "str | None":
    """The user's chosen label for an account (by internal key), or None if unlabeled."""
    return load_account_labels().get(resolve_account(account) or primary_account())


def set_account_label(account: "str | None", label: str) -> str:
    """Assign/update the display label for an account. `account` may be a key, email, or
    existing label. Returns the internal key it was set on."""
    key = resolve_account(account) or primary_account()
    labels = load_account_labels()
    labels[key] = label.strip()
    _save_account_labels(labels)
    return key


def clear_account_label(account: "str | None") -> str:
    """Remove an account's label (revert to showing its email). Returns the internal key."""
    key = resolve_account(account) or primary_account()
    labels = load_account_labels()
    labels.pop(key, None)
    _save_account_labels(labels)
    return key


def account_display(account: "str | None") -> str:
    """How to refer to an account in conversation: its label if set, else its email,
    else the internal key as a last resort."""
    key = resolve_account(account) or primary_account()
    return load_account_labels().get(key) or account_email(key) or key


def resolve_account(value: "str | None") -> "str | None":
    """Map a user-facing account identifier — an internal key, a full email address, OR a
    user-assigned display label (all case-insensitive) — to the internal key used for
    token/state files. None and the 'all' sentinel pass through unchanged; an unknown
    value is returned as-is so the caller surfaces a clear 'not connected' error."""
    if not value:
        return value
    v = value.strip()
    if v.lower() in {"all", "every", "everything", "both"}:
        return v
    avail = available_accounts()
    for a in avail:                                   # already an internal key?
        if a and a.lower() == v.lower():
            return a
    for key, label in load_account_labels().items():  # a user-assigned label?
        if label.lower() == v.lower() and key in avail:
            return key
    if "@" in v:                                      # an email address — find its key
        for a in avail:
            em = account_email(a)
            if em and em.lower() == v.lower():
                return a
    return value


def accounts_for(account: "str | None") -> list:
    """Resolve a request's account arg: None/'all' -> every authorized account; a
    specific label OR email address -> just that one (if authorized, else empty)."""
    if account and account.lower() != "all":
        account = resolve_account(account)            # accept an email or a label
        return [account] if os.path.exists(_token_path(account)) else []
    return available_accounts()


def _granted_scopes(path: str) -> set:
    import json
    try:
        with open(path, "r", encoding="utf-8") as f:
            return set(json.load(f).get("scopes", []) or [])
    except (OSError, ValueError):
        return set()


def _save(creds, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(creds.to_json())


def _credentials(account: str, interactive: bool = False):
    """Authorized OAuth credentials for `account`, covering config.GOOGLE_SCOPES.

    interactive=False (normal tool/service use): if there's no valid token, RAISE —
    never open a browser from a tool call or the background scanner. interactive=True
    (explicit authorize/setup): run the consent browser flow when needed.
    """
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request

    scopes = config.GOOGLE_SCOPES
    path = _token_path(account)
    creds = None
    # Reuse the saved token only if it grants every scope we need (else re-consent —
    # Google rejects refreshing a token up to broader scopes as invalid_scope).
    if os.path.exists(path) and set(scopes).issubset(_granted_scopes(path)):
        creds = Credentials.from_authorized_user_file(path, scopes)
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        _save(creds, path)
        return creds
    if not interactive:
        raise RuntimeError(
            f"Google account '{account}' isn't connected — authorize it first "
            "(ask me to connect it, or run google_auth.py).")
    if not os.path.exists(config.GOOGLE_CREDENTIALS_PATH):
        raise FileNotFoundError(
            f"missing {config.GOOGLE_CREDENTIALS_PATH} — download an OAuth "
            "'desktop app' credentials.json from the Google Cloud console first")
    log.debug("consenting account %s for scopes: %s", account, scopes)
    flow = InstalledAppFlow.from_client_secrets_file(config.GOOGLE_CREDENTIALS_PATH, scopes)
    creds = flow.run_local_server(port=0)
    _save(creds, path)
    return creds


def service(api: str, version: str, account: "str | None" = None):
    """Authenticated Google API client for an account (defaults to the primary). Never
    opens a browser — raises if the account isn't connected."""
    account = resolve_account(account) or primary_account()  # accept a label or an email
    key = (api, version, account)
    if key not in _services:
        from googleapiclient.discovery import build
        _services[key] = build(api, version, credentials=_credentials(account, interactive=False),
                               cache_discovery=False)
    return _services[key]


if __name__ == "__main__":
    acct = sys.argv[1] if len(sys.argv) > 1 else primary_account()
    print(f"Authorizing Google account '{acct}' for scopes:\n  " + "\n  ".join(config.GOOGLE_SCOPES))
    print("Sign in with the CORRECT Google account when the browser opens…")
    _credentials(acct, interactive=True)
    print(f"Done — token saved to {_token_path(acct)}")
