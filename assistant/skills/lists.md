---
name: lists
description: Create and manage named lists (groceries, todos, packing) — add, remove, show, list all.
triggers: (add|put|cross off|take off|remove|delete).*(list|todo)|(grocery|shopping|to-?do|packing|wish|reading|bucket).*list|(show|read|what'?s on|whats on).*list|my lists|(create|start|make).*list|list called|on my .* list
---
When I talk about my lists, use the list tools — never store list items in plain memory:

- **"add X to my Y list" / "put X on the Y list"** → `add_to_list(name="Y", items="X")`. It
  creates the list if it doesn't exist yet, so you don't need a separate create step.
- **"create/start a Y list"** (optionally "with A, B, C") → `create_list(name="Y", items="A, B, C")`.
- **"remove/cross off X from my Y list"** → `remove_from_list(name="Y", items="X")`.
- **"show / what's on my Y list" / "read me the Y list"** → `show_list(name="Y")`.
- **"what lists do I have" / "show my lists"** → `list_lists()` (returns each list's name,
  item count, and created/updated dates).
- **"rename the Y list to Z"** → `rename_list`. **"delete the whole Y list"** → `delete_list`
  (only for deleting an ENTIRE list, not single items — use remove_from_list for items).

Notes:
- The list NAME is what I call it ("groceries", "apartment shopping"); it's matched
  case-insensitively, so don't fuss about exact capitalization.
- Multiple items in one go: pass them comma- or newline-separated in `items`.
- After acting, briefly confirm what changed (e.g. "Added milk and eggs — groceries now has
  6 items"). For show/list, relay the tool's output as-is (it's already formatted).
