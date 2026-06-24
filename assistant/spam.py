"""Spam-candidate log + the periodic scan that fills it.

The scan is read-only: it flags senders with many unread emails and records them
to config.SPAM_LOG_PATH. NOTHING is deleted here — the user reviews and confirms
each sender through the spam-cleanup flow. The in-app scanner (started from
main()) runs every config.SPAM_SCAN_INTERVAL seconds (default 6 hours) while Karl
is open.
"""
import json
import logging
import os
import sys
import threading
import time

# Allow running directly for an on-the-spot scan: `python assistant/spam.py`.
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

log = logging.getLogger("assistant.spam")


def _write_json(path: str, data) -> None:
    """Atomically write JSON (temp file + rename) so a crash mid-write can't truncate a
    file — important for the keep-list / auto-delete safety lists."""
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except OSError as e:
        log.debug("could not write %s: %s", path, e)
        try:
            os.unlink(tmp)
        except OSError:
            pass


# --- per-account state files -------------------------------------------------
# Each connected account keeps its OWN candidate log, keep-list, auto-delete list, and
# scan checkpoint, so e.g. a sender you keep in 'personal' is never auto-trashed in
# 'work'. The PRIMARY account uses the base paths (backward compatible with existing
# files); every other account gets an '_<account>' suffix before the extension.
def _primary() -> "str | None":
    return config.GOOGLE_ACCOUNTS[0] if config.GOOGLE_ACCOUNTS else None


def _p(base_path: str, account: "str | None") -> str:
    if not account or account == _primary():
        return base_path
    root, ext = os.path.splitext(base_path)
    return f"{root}_{account}{ext}"


def record_candidates(candidates: list, account: str = None) -> None:
    """Persist the latest scan result (timestamped) to the account's log file."""
    _write_json(_p(config.SPAM_LOG_PATH, account),
                {"ts": time.time(), "candidates": candidates})


def _load(account: str = None) -> dict:
    try:
        with open(_p(config.SPAM_LOG_PATH, account), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def load_keep(account: str = None) -> set:
    """Return the set of kept (never-spam) senders/domains, lowercased."""
    try:
        with open(_p(config.SPAM_KEEP_PATH, account), "r", encoding="utf-8") as f:
            return {s.strip().lower() for s in json.load(f) if s.strip()}
    except (OSError, ValueError):
        return set()


def add_keep(senders, account: str = None) -> set:
    """Add one or more senders/domains to the keep-list. Returns the new set."""
    if isinstance(senders, str):
        senders = [senders]
    keep = load_keep(account)
    keep.update(s.strip().lower() for s in senders if s and s.strip())
    _write_json(_p(config.SPAM_KEEP_PATH, account), sorted(keep))
    return keep


def matches(sender: str, entries: set) -> bool:
    """True if `sender` matches a list entry — by exact address, exact domain, or a
    parent domain (so a 'regenics.com' entry also covers 'send.regenics.com')."""
    if not entries:
        return False
    s = (sender or "").lower()
    if s in entries:
        return True
    dom = s.split("@")[-1]
    return any(dom == e or dom.endswith("." + e) for e in entries)


def is_kept(sender: str, keep: set = None) -> bool:
    """True if `sender` (or its domain) is on the keep-list."""
    return matches(sender, keep if keep is not None else load_keep())


def load_autodelete(account: str = None) -> set:
    """Senders/domains the user confirmed as junk — auto-trashed each scan."""
    try:
        with open(_p(config.SPAM_AUTODELETE_PATH, account), "r", encoding="utf-8") as f:
            return {s.strip().lower() for s in json.load(f) if s.strip()}
    except (OSError, ValueError):
        return set()


def add_autodelete(senders, account: str = None) -> set:
    """Add sender(s)/domain(s) to the auto-delete list. Returns the new set."""
    if isinstance(senders, str):
        senders = [senders]
    block = load_autodelete(account)
    block.update(s.strip().lower() for s in senders if s and s.strip())
    _write_json(_p(config.SPAM_AUTODELETE_PATH, account), sorted(block))
    return block


def is_autodelete(sender: str, block: set = None) -> bool:
    return matches(sender, block if block is not None else load_autodelete())


def load_candidates(account: str = None) -> list:
    """Most recent scan's candidates, with kept and auto-delete senders filtered out
    (so a stale log never resurfaces a sender you've since chosen to keep or auto-delete)."""
    keep, block = load_keep(account), load_autodelete(account)
    return [c for c in (_load(account).get("candidates", []) or [])
            if not is_kept(c.get("sender", ""), keep)
            and not is_autodelete(c.get("sender", ""), block)]


def last_scan_age(account: str = None) -> "float | None":
    """Seconds since the last recorded scan, or None if there's never been one."""
    ts = _load(account).get("ts")
    return (time.time() - ts) if ts else None


# --- progress announcements (set by main() so updates print and/or speak) -----
_announcer = None


def set_announcer(fn) -> None:
    """Register how batch-progress updates reach the user (print and/or speak)."""
    global _announcer
    _announcer = fn


def announce(msg: str) -> None:
    log.info(msg)
    try:
        (_announcer or (lambda m: print(f"  📬 {m}")))(msg)
    except Exception as e:  # noqa: BLE001 — a progress update must never break the scan
        log.debug("announce failed: %s", e)


# --- resumable scan checkpoint (so a huge scan survives interruption) ----------
def save_scan_state(state: dict, account: str = None) -> None:
    _write_json(_p(config.SPAM_SCAN_STATE_PATH, account), state)


def load_scan_state(account: str = None) -> dict:
    try:
        with open(_p(config.SPAM_SCAN_STATE_PATH, account), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def clear_scan_state(account: str = None) -> None:
    try:
        os.unlink(_p(config.SPAM_SCAN_STATE_PATH, account))
    except OSError:
        pass


def connected_accounts() -> list:
    """Connected Google accounts to scan/clean (primary first); [None] if none, so a
    single-account setup still works with the base state files."""
    try:
        from tools import google_auth
        return google_auth.available_accounts() or [None]
    except Exception as e:  # noqa: BLE001
        log.debug("account discovery failed: %s", e)
        return [None]


def run_scan(max_scan: int = None, account: str = None) -> list:
    """Scan now (live Gmail): auto-trash unread from confirmed auto-delete senders,
    then record the remaining candidates for review. Skips kept + auto-delete senders.
    Pass max_scan=0 to scan the ENTIRE unread history (slower). Returns the candidates."""
    from tools import gmail_tool
    block = load_autodelete(account)
    auto_n = 0
    if block:
        try:
            auto_n, _ = gmail_tool.auto_trash_blocked(block, account=account)
        except Exception as e:  # noqa: BLE001
            log.debug("auto-trash failed: %s", e)
    cands = gmail_tool.find_spam_candidates(
        max_scan=max_scan, exclude=load_keep(account) | block, account=account)
    record_candidates(cands, account)
    log.info("spam scan [%s]: auto-trashed %d, %d sender(s) to review",
             account or "primary", auto_n, len(cands))
    return cands


def start_background_scanner() -> None:
    """Start the periodic scanner in a daemon thread (read-only; logs only). Scans EVERY
    connected account each cycle."""
    if not (config.GMAIL_ENABLED and config.SPAM_SCAN_ENABLED):
        return

    def _loop():
        # Skip the immediate scan if a recent one already exists for the primary (avoids
        # hammering Gmail when Karl is relaunched often).
        age = last_scan_age()
        if age is not None and age < config.SPAM_SCAN_INTERVAL:
            time.sleep(config.SPAM_SCAN_INTERVAL - age)
        while True:
            for acct in connected_accounts():
                try:
                    run_scan(account=acct)
                except Exception as e:  # noqa: BLE001 — never let the scanner crash Karl
                    log.debug("background spam scan [%s] failed: %s", acct or "primary", e)
            time.sleep(config.SPAM_SCAN_INTERVAL)

    threading.Thread(target=_loop, name="spam-scanner", daemon=True).start()
    log.debug("spam scanner started (every %ds, all accounts)", config.SPAM_SCAN_INTERVAL)


if __name__ == "__main__":
    # On-the-spot scan: report senders with many unread and record them.
    # Pass --full to scan the ENTIRE unread history (no sample cap; slower).
    full = "--full" in sys.argv
    where = "ALL unread mail" if full else f"the first {config.SPAM_SCAN_MAX} unread"
    print(f"Scanning {where} for senders with more than "
          f"{config.SPAM_UNREAD_THRESHOLD} unread…")
    cands = run_scan(max_scan=0 if full else None)
    if not cands:
        print("No spam candidates found.")
    else:
        print(f"\n{len(cands)} sender(s) over the threshold:")
        for c in cands:
            unsub = "  (can unsubscribe)" if c.get("unsubscribe") else ""
            print(f"  - {c['sender']}: {c['count']} unread{unsub}")
        print('\nRun "spam cleanup" in Karl to review and delete/unsubscribe.')
