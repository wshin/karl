"""Named lists Karl can create, view, and edit — groceries, todos, packing, …

Each list is one JSON file under config.LISTS_DIR: {name, items, created, updated}. The
list's display name maps to a slugified filename ('My Groceries' -> my-groceries.json);
lookups match the slug, then fall back to a case-insensitive stored-name match. All ops
return a short human-readable string for the model to relay.
"""
import datetime
import json
import logging
import os
import re

import config

log = logging.getLogger("assistant.lists")


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return s or "list"


def _path(slug: str) -> str:
    return os.path.join(config.LISTS_DIR, slug + ".json")


def _all() -> list:
    """Every stored list (dicts), newest-updated first."""
    out = []
    try:
        for fn in os.listdir(config.LISTS_DIR):
            if fn.endswith(".json"):
                try:
                    with open(os.path.join(config.LISTS_DIR, fn), encoding="utf-8") as f:
                        out.append(json.load(f))
                except (OSError, ValueError):
                    pass
    except OSError:
        pass
    out.sort(key=lambda d: d.get("updated", ""), reverse=True)
    return out


def _find(name: str) -> "dict | None":
    """Find a list by name: exact slug file first, then a case-insensitive name match."""
    p = _path(_slug(name))
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            pass
    target = (name or "").strip().lower()
    for d in _all():
        if d.get("name", "").strip().lower() == target:
            return d
    return None


def _save(data: dict) -> dict:
    os.makedirs(config.LISTS_DIR, exist_ok=True)
    data["updated"] = _now()
    path = _path(_slug(data["name"]))
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except OSError as e:
        log.debug("could not write list %s: %s", data.get("name"), e)
    return data


def _items(items) -> list:
    """Normalize items given as a list, or a comma/newline-separated string."""
    raw = items if isinstance(items, list) else re.split(r"[\n,]", items or "")
    return [str(i).strip() for i in raw if str(i).strip()]


def create_list(name: str, items: str = None) -> str:
    """Create a new named list (optionally with initial items)."""
    name = (name or "").strip()
    if not name:
        return "ERROR: give the list a name."
    if _find(name):
        return f"A list called '{name}' already exists — add to it or show it instead."
    data = _save({"name": name, "items": _items(items), "created": _now(), "updated": _now()})
    n = len(data["items"])
    return f"Created the list '{name}'" + (f" with {n} item(s)." if n else " (empty).")


def add_to_list(name: str, items: str) -> str:
    """Add item(s) to a list (comma- or newline-separated). Creates the list if new."""
    new = _items(items)
    if not new:
        return "ERROR: tell me what to add."
    data = _find(name)
    created = data is None
    if created:
        data = {"name": (name or "").strip(), "items": [], "created": _now(), "updated": _now()}
    have = {i.lower() for i in data["items"]}
    added = [i for i in new if i.lower() not in have]
    data["items"].extend(added)
    _save(data)
    dup = len(new) - len(added)
    head = f"Created '{data['name']}' and added" if created else f"Added"
    extra = f" ({dup} already on it)" if dup else ""
    return f"{head} {len(added)} item(s) to '{data['name']}'{extra}. Now {len(data['items'])} item(s)."


def remove_from_list(name: str, items: str) -> str:
    """Remove item(s) from a list (matched case-insensitively)."""
    data = _find(name)
    if not data:
        return f"No list called '{name}'. Say 'show my lists' to see what you have."
    drop = {i.lower() for i in _items(items)}
    before = len(data["items"])
    data["items"] = [i for i in data["items"] if i.lower() not in drop]
    removed = before - len(data["items"])
    _save(data)
    if not removed:
        return f"Nothing matched in '{data['name']}' (still {len(data['items'])} item(s))."
    return f"Removed {removed} item(s) from '{data['name']}'. Now {len(data['items'])} item(s)."


def show_list(name: str) -> str:
    """Show one list's items (numbered) with its item count."""
    data = _find(name)
    if not data:
        return f"No list called '{name}'. Say 'show my lists' to see what you have."
    items = data.get("items", [])
    if not items:
        return f"'{data['name']}' is empty."
    body = "\n".join(f"  {i}. {it}" for i, it in enumerate(items, 1))
    return f"{data['name']} ({len(items)} item(s)):\n{body}"


def list_lists() -> str:
    """List every list by name, with item count and created/updated dates."""
    lists = _all()
    if not lists:
        return "You don't have any lists yet. Say 'create a list called X' to start one."
    lines = []
    for i, d in enumerate(lists, 1):
        created = (d.get("created", "") or "")[:10]
        updated = (d.get("updated", "") or "")[:10]
        lines.append(f"  {i}. {d.get('name', '?')} — {len(d.get('items', []))} item(s)"
                     f"  (created {created or '?'}, updated {updated or '?'})")
    return f"You have {len(lists)} list(s):\n" + "\n".join(lines)


def delete_list(name: str) -> str:
    """Delete a named list entirely."""
    data = _find(name)
    if not data:
        return f"No list called '{name}'."
    try:
        os.unlink(_path(_slug(data["name"])))
    except OSError as e:
        return f"Couldn't delete '{data['name']}': {e}"
    return f"Deleted the list '{data['name']}' (had {len(data.get('items', []))} item(s))."


def rename_list(name: str, new_name: str) -> str:
    """Rename a list, keeping its items and created date."""
    data = _find(name)
    if not data:
        return f"No list called '{name}'."
    new_name = (new_name or "").strip()
    if not new_name:
        return "ERROR: give the new name."
    if _slug(new_name) != _slug(data["name"]) and _find(new_name):
        return f"A list called '{new_name}' already exists."
    old_path = _path(_slug(data["name"]))
    data["name"] = new_name
    _save(data)
    if _path(_slug(new_name)) != old_path:
        try:
            os.unlink(old_path)
        except OSError:
            pass
    return f"Renamed the list to '{new_name}'."


_NAME = {"type": "string", "description": "The list's name (case-insensitive)."}
_ITEMS = {"type": "string", "description": "Item(s), comma- or newline-separated."}

LIST_SCHEMAS = [
    {"type": "function", "function": {
        "name": "create_list",
        "description": "Create a new named list (e.g. groceries, todo, packing). Optionally "
                       "seed it with items. Use when the user asks to start/make a new list.",
        "parameters": {"type": "object",
                       "properties": {"name": _NAME, "items": {"type": "string", "description": "Optional initial items, comma/newline-separated."}},
                       "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "add_to_list",
        "description": "Add one or more items to a named list (creates the list if it doesn't "
                       "exist). Use for 'add X to my Y list', 'put X on the Y list'.",
        "parameters": {"type": "object", "properties": {"name": _NAME, "items": _ITEMS},
                       "required": ["name", "items"]}}},
    {"type": "function", "function": {
        "name": "remove_from_list",
        "description": "Remove one or more items from a named list. Use for 'remove X from my "
                       "Y list', 'cross off X', 'take X off the Y list'.",
        "parameters": {"type": "object", "properties": {"name": _NAME, "items": _ITEMS},
                       "required": ["name", "items"]}}},
    {"type": "function", "function": {
        "name": "show_list",
        "description": "Show the items on a single named list. Use for 'show/what's on my Y "
                       "list', 'read me the Y list'.",
        "parameters": {"type": "object", "properties": {"name": _NAME}, "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "list_lists",
        "description": "List ALL of the user's lists by name, with each list's item count and "
                       "created/updated dates. Use for 'what lists do I have', 'show my lists'.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "rename_list",
        "description": "Rename an existing list, keeping its items.",
        "parameters": {"type": "object",
                       "properties": {"name": _NAME, "new_name": {"type": "string", "description": "The new name."}},
                       "required": ["name", "new_name"]}}},
    {"type": "function", "function": {
        "name": "delete_list",
        "description": "Delete an entire named list. Use only when the user clearly asks to "
                       "delete/remove a whole list (not just items from it).",
        "parameters": {"type": "object", "properties": {"name": _NAME}, "required": ["name"]}}},
]

LIST_FUNCTIONS = {
    "create_list": create_list,
    "add_to_list": add_to_list,
    "remove_from_list": remove_from_list,
    "show_list": show_list,
    "list_lists": list_lists,
    "rename_list": rename_list,
    "delete_list": delete_list,
}
