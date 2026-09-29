#!/usr/bin/env python3
"""Vendor a curated subset of the DIVE manual into the VIAME Sphinx build.

DIVE authors its docs for mkdocs-material; this rewrites the material-only
syntax into MyST so the pages build as part of the VIAME manual.
"""

import hashlib
import os
import posixpath
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUTPUT = HERE / "sections" / "dive"
CHECKOUT = HERE / "_dive_src"
STAMP = OUTPUT / ".stamp"

DIVE_REMOTE = "https://github.com/Kitware/dive.git"
DIVE_SITE = "https://kitware.github.io/dive"
SUBMODULE = "packages/dive"

ENTRY = "index.md"
ENTRY_TITLE = "DIVE Interface"

NAV_FILE = "mkdocs.yml"
GROUP_PREFIX = "menu-"

NAV = []
PAGES = {}

ADMONITION_CLASS = {
    "note": "note",
    "info": "note",
    "abstract": "note",
    "example": "note",
    "question": "hint",
    "tip": "tip",
    "success": "tip",
    "warning": "warning",
    "caution": "caution",
    "danger": "danger",
    "failure": "error",
    "bug": "error",
    "quote": "note",
}

FENCE = re.compile(r"^\s*(```|~~~)")
ADMONITION = re.compile(r'^(!!!|\?\?\?\+?)\s+(\w+)(?:\s+"([^"]*)")?\s*(.*)$')
ICON = re.compile(r":(?:material|fontawesome|octicons)-([a-z0-9-]+):")
KEYS = re.compile(r"\+\+([A-Za-z0-9-]+(?:\+[A-Za-z0-9-]+)*)\+\+")
FA_STYLES = ("solid-", "regular-", "brands-", "light-")
MARK = re.compile(r"==\s*([^=\n]+?)\s*==")
ATTR_LIST = re.compile(r"\{\s*[.#][^}\n]*\}")
MD_LINK = re.compile(r"\]\((?!https?:|/|#)([A-Za-z0-9._/-]+\.md)(#[A-Za-z0-9._-]+)?\)")
LOGOS = re.compile(r"<p>\s*(?:<img[^>]*>\s*)+</p>\n*")
BUTTON = "{ .md-button }"
ASSET = re.compile(r"(?:\]\(|src=[\"'])((?:\.\./)*(?:images|videos)/[^)\"'\s]+)")


class NavLoader(yaml.SafeLoader):
    pass


NavLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/", lambda loader, suffix, node: None)


def log(message):
    print("[dive-docs] " + message, file=sys.stderr)


def dive_ref():
    ref = os.environ.get("VIAME_DIVE_DOCS_REF")
    if ref:
        return ref
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "ls-tree", "HEAD", SUBMODULE],
            capture_output=True, text=True, timeout=30, check=True).stdout.split()
        if len(out) >= 3:
            return out[2]
    except (subprocess.SubprocessError, OSError):
        pass
    return "main"


def fetch(ref):
    """Partial sparse fetch; the full DIVE submodule is >2GB, the docs are 48MB."""
    CHECKOUT.mkdir(parents=True, exist_ok=True)
    git = ["git", "-C", str(CHECKOUT)]
    steps = [
        git + ["init", "-q"],
        git + ["remote", "add", "origin", DIVE_REMOTE],
        git + ["fetch", "-q", "--filter=blob:none", "--depth", "1", "origin", ref],
        git + ["sparse-checkout", "init", "--cone"],
        git + ["sparse-checkout", "set", "docs"],
        git + ["checkout", "-q", "FETCH_HEAD"],
    ]
    for step in steps:
        result = subprocess.run(step, capture_output=True, text=True, timeout=600)
        if result.returncode != 0 and "remote add" not in " ".join(step):
            log("fetch failed: " + result.stderr.strip())
            return None
    docs = CHECKOUT / "docs"
    return docs if (docs / "index.md").exists() else None


def source_docs(ref):
    local = REPO / SUBMODULE / "docs"
    if (local / "index.md").exists():
        log("using checked-out submodule at " + str(local))
        return local
    log("fetching DIVE docs at " + ref[:12])
    return fetch(ref)


def nav_entries(items):
    entries = []
    for item in items:
        title, target = (None, item) if isinstance(item, str) else next(iter(item.items()))
        if isinstance(target, list):
            entries.append((title, nav_entries(target)))
        elif target != ENTRY:
            entries.append((title or Path(target).stem, target))
    return entries


def nav_pages(entries):
    pages = {}
    for title, target in entries:
        if isinstance(target, list):
            pages.update(nav_pages(target))
        elif target.endswith(".md") and "://" not in target:
            pages[target] = title
    return pages


def load_nav(source):
    """The menu is DIVE's own, so this section is laid out like the DIVE site."""
    config = yaml.load(
        (source.parent / NAV_FILE).read_text(encoding="utf-8"), Loader=NavLoader)
    NAV[:] = nav_entries(config.get("nav") or [])
    PAGES.clear()
    PAGES.update(nav_pages(NAV))


def capitalize_key(token):
    if token.startswith("arrow-"):
        return token[len("arrow-"):].capitalize() + " Arrow"
    token = token.replace("-", " ")
    return token.upper() if len(token) == 1 else token.title()


def icon_text(match):
    """Icons carry meaning ("click the cog"), so name them rather than drop them."""
    name = match.group(1)
    for style in FA_STYLES:
        if name.startswith(style):
            name = name[len(style):]
    name = name.replace("-", " ")
    return name if name.endswith("icon") else name + " icon"


def convert_inline(line, page):
    line = ICON.sub(icon_text, line)
    line = ATTR_LIST.sub("", line)
    line = KEYS.sub(lambda m: "`" + "+".join(
        capitalize_key(t) for t in m.group(1).split("+")) + "`", line)
    line = MARK.sub(r"**\1**", line)
    line = MD_LINK.sub(lambda match: rewrite_link(match, page), line)
    line = re.sub(r"\[\s+", "[", line)
    return line.rstrip()


def rewrite_link(match, page):
    target, anchor = match.group(1), match.group(2) or ""
    folder = posixpath.dirname(page)
    resolved = posixpath.normpath(posixpath.join(folder, target))
    # Pages in subfolders link some top-level pages as if they sat beside them
    for name in (resolved, posixpath.basename(target)):
        if name in PAGES or name == ENTRY:
            return "](" + posixpath.relpath(name, folder or ".") + anchor + ")"
    return "](" + DIVE_SITE + "/" + resolved[:-3] + "/" + anchor + ")"


def strip_front_matter(lines):
    if lines and lines[0].strip() == "---":
        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                return lines[index + 1:]
    return lines


def expand_admonitions(lines):
    out, index = [], 0
    in_fence = False
    while index < len(lines):
        line = lines[index]
        if FENCE.match(line):
            in_fence = not in_fence
        match = None if in_fence else ADMONITION.match(line)
        if not match:
            out.append(line)
            index += 1
            continue
        _, kind, title, trailing = match.groups()
        body = [trailing] if trailing.strip() else []
        index += 1
        while index < len(lines):
            nxt = lines[index]
            if nxt.strip() and not nxt.startswith("    "):
                break
            body.append(nxt[4:] if nxt.startswith("    ") else nxt)
            index += 1
        while body and not body[0].strip():
            body.pop(0)
        while body and not body[-1].strip():
            body.pop()
        css = ADMONITION_CLASS.get(kind.lower(), "note")
        out.append(":::{admonition} " + (title or kind.capitalize()))
        out.append(":class: " + css)
        out.append("")
        out.extend(expand_admonitions(body))
        out.append(":::")
        out.append("")
    return out


def convert(text, title, page):
    lines = strip_front_matter(text.splitlines())
    lines = expand_admonitions(lines)
    out, in_fence = [], False
    for line in lines:
        if FENCE.match(line):
            in_fence = not in_fence
            out.append(line.replace("```mermaid", "```text"))
            continue
        out.append(line if in_fence else convert_inline(line, page))
    body = "\n".join(out).strip("\n")
    if not re.search(r"^# ", body, re.M):
        body = "# " + title + "\n\n" + body
    return body + "\n"


def entry_text(source):
    """The opening page of the DIVE manual, worded for its place in this one."""
    text = (source / ENTRY).read_text(encoding="utf-8")
    text = re.sub(r"^# .*$", "# " + ENTRY_TITLE, text, count=1, flags=re.M)
    return text.replace("This is the documentation site for DIVE, a", "DIVE is a", 1)


def page_assets(text, page, source):
    folder = posixpath.dirname(page)
    paths = {posixpath.normpath(posixpath.join(folder, path)) for path in ASSET.findall(text)}
    return sorted(path for path in paths if (source / path).is_file())


def copy_assets(text, page, source, dest):
    for path in page_assets(text, page, source):
        origin = source / path
        target = dest / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target)


def toctree(entries):
    lines = ["```{toctree}", ":maxdepth: 1", ""]
    for title, target in entries:
        if isinstance(target, list):
            target = write_group(title, target)
        elif target.endswith(".md"):
            target = target[:-3]
        lines.append(title + " <" + target + ">")
    return lines + ["```", ""]


def write_group(title, entries):
    name = GROUP_PREFIX + re.sub(r"[^A-Za-z0-9]+", "-", title).strip("-")
    lines = ["# " + title, ""] + toctree(entries)
    (OUTPUT / (name + ".md")).write_text("\n".join(lines), encoding="utf-8")
    return name


def write_index(ref, source):
    entry = ICON.sub("", entry_text(source))
    entry = LOGOS.sub("", entry).replace(BUTTON + " [", BUTTON + " | [")
    copy_assets(entry, ENTRY, source, OUTPUT)
    lines = [
        convert(entry, ENTRY_TITLE, ENTRY),
        "The pages below are vendored from the [DIVE manual](" + DIVE_SITE + ") at the",
        "revision VIAME currently ships (`" + ref[:12] + "`). The upstream site is",
        "authoritative for anything newer.",
        "",
    ] + toctree(NAV)
    (OUTPUT / "index.md").write_text("\n".join(lines), encoding="utf-8")


def write_placeholder(reason):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "index.rst").write_text(
        ENTRY_TITLE + "\n" + "=" * len(ENTRY_TITLE) + "\n\n"
        "The DIVE manual could not be retrieved for this build (" + reason + ").\n"
        "See `the DIVE documentation site <" + DIVE_SITE + ">`_.\n",
        encoding="utf-8")


def fingerprint(ref):
    digest = hashlib.sha256()
    digest.update(ref.encode())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def generate(force=False):
    ref = dive_ref()
    stamp = fingerprint(ref)
    if not force and STAMP.exists() and STAMP.read_text().strip() == stamp:
        return True
    source = source_docs(ref)
    if source is None:
        write_placeholder("DIVE checkout unavailable")
        return False
    load_nav(source)
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    count = 0
    for name, title in PAGES.items():
        origin = source / name
        if not origin.is_file():
            log("skipping missing page " + name)
            continue
        text = convert(origin.read_text(encoding="utf-8"), title, name)
        (OUTPUT / name).parent.mkdir(parents=True, exist_ok=True)
        (OUTPUT / name).write_text(text, encoding="utf-8")
        copy_assets(text, name, source, OUTPUT)
        count += 1
    write_index(ref, source)
    STAMP.write_text(stamp, encoding="utf-8")
    log("converted {0} pages into {1}".format(count, OUTPUT))
    return True


if __name__ == "__main__":
    sys.exit(0 if generate(force="--force" in sys.argv) else 1)
