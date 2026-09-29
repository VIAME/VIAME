"""Build the VIAME manual with MkDocs from the sources the Sphinx build uses.

Pages are converted from RST in memory on every build and handed to MkDocs as
generated files, so nothing is written into the source tree and the Sphinx
build is unaffected. The navigation is read from the toctree in index.rst.
"""

import logging
import posixpath
import re
import sys
from pathlib import Path

import pypandoc
from mkdocs.structure.files import File

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dive_docs  # noqa: E402

log = logging.getLogger("mkdocs.hooks.viame")

SECTIONS = "sections"
DIVE_DIR = SECTIONS + "/dive"
DIVE_INDEX = DIVE_DIR + "/index"

INCLUDE = re.compile(r"^\.\. include:: *(\S+) *\n((?:[ \t]+:[\w-]+:.*\n)*)", re.M)
TOCTREE = re.compile(r"^\.\. toctree::.*\n((?:(?:[ \t]+.*)?\n)*)", re.M)
TOCTREE_HEADING = re.compile(
    r"^[^\n]+\n([=\-~^\"*+#])\1{2,} *\n\s*(?=\.\. toctree::)", re.M)
TITLED_ENTRY = re.compile(r"^(.*?)\s*<([^<>]+)>$")
RTD = r"https?://viame\.readthedocs\.io/en/latest/?([^\s)#\"'>]*)(#[^\s)\"'>]*)?"
RTD_LINK = re.compile(RTD)
RTD_AUTOLINK = re.compile("<" + RTD + ">")
RTD_TARGET = re.compile(r"\]\(" + RTD + r"\)")
FENCE = re.compile(r"^(```+|~~~+)")
FENCE_LANGUAGE = re.compile(r"^(```+) +(\S)", re.M)
HEADING = re.compile(r"^(#+) +(.*)$")
ALERT = re.compile(r"^> \[!(\w+)\] *$")
IMAGE = re.compile(r"<img ([^>]*?) */>")
IMAGE_ROW = re.compile(
    r"^(?:\[?!\[[^\]]*\]\([^)]*\)(?:\{[^}]*\})?(?:\]\([^)]*\))? ?)+$")
ATTRIBUTE = re.compile(r"([\w-]+)=\"([^\"]*)\"")
DIV = re.compile(r"^<div\b([^>]*)>$")
PAREN_LIST = re.compile(r"^(\s*\d+)\)(\s)")

_generated = {}


def flatten(path, text=None):
    """Expand include directives the way docutils would."""
    def expand(match):
        target = (path.parent / match.group(1)).resolve()
        options = dict(re.findall(r":([\w-]+): *(.*)", match.group(2)))
        body = flatten(target)
        if "start-after" in options:
            body = body.partition(options["start-after"])[2]
        if "end-before" in options:
            body = body.partition(options["end-before"])[0]
        return body + "\n"
    if text is None:
        text = path.read_text(encoding="utf-8")
    return INCLUDE.sub(expand, text)


def toctree_entries(text):
    """Yield (title, target) for every toctree entry, title may be None."""
    for block in TOCTREE.findall(text):
        for line in block.splitlines():
            line = line.strip()
            if not line or line.startswith(":"):
                continue
            match = TITLED_ENTRY.match(line)
            yield (match.group(1), match.group(2)) if match else (None, line)


def strip_toctrees(text):
    return TOCTREE.sub("", TOCTREE_HEADING.sub("", text))


def outside_fences(lines, skip_tables=False):
    in_fence = in_table = False
    for index, line in enumerate(lines):
        if FENCE.match(line):
            in_fence = not in_fence
        elif skip_tables and line.startswith(("<table", "</table")):
            in_table = line.startswith("<table")
        elif not in_fence and not in_table:
            yield index, line


def fix_headings(lines, fallback_title):
    """Material builds the page outline from a single top-level heading."""
    headings = [(i, HEADING.match(l)) for i, l in outside_fences(lines)]
    headings = [(i, len(m.group(1))) for i, m in headings if m]
    top = [i for i, level in headings if level == 1]
    if len(top) == 1:
        return lines
    keep = None
    if len(top) > 1 and len(headings) > 1 and headings[1][1] == 1:
        keep = top[0]
    for index, _ in headings:
        if index != keep:
            lines[index] = "#" + lines[index]
    if keep is None:
        lines[:0] = ["# " + fallback_title, ""]
    return lines


def convert_alerts(lines):
    out, index = [], 0
    while index < len(lines):
        match = ALERT.match(lines[index])
        if not match:
            out.append(lines[index])
            index += 1
            continue
        out.append("!!! " + match.group(1).lower())
        out.append("")
        index += 1
        while index < len(lines) and lines[index].startswith(">"):
            out.append("    " + lines[index][1:].lstrip(" "))
            index += 1
    return out


def convert_image(match):
    attributes = dict(ATTRIBUTE.findall(match.group(1)))
    extra = ["." + name for name in attributes.get("class", "").split()]
    if attributes.get("style"):
        style = re.sub(r"(\d+)\.0%", r"\1%", attributes["style"])
        extra.append('style="' + style + '"')
    image = "![{0}]({1})".format(attributes.get("alt", ""), attributes.get("src", ""))
    return image + "{ " + " ".join(extra) + " }" if extra else image


def is_image_row(line):
    return bool(IMAGE_ROW.match(line)) and ".align-center" not in line


def join_image_rows(lines):
    """Sphinx flows consecutive images onto one row; keep them in one paragraph."""
    out = []
    for line in lines:
        if (is_image_row(line) and len(out) >= 2 and not out[-1] and
                is_image_row(out[-2])):
            out.pop()
            out[-1] += " " + line
        else:
            out.append(line)
    return out


def relink(text, page, pages):
    """Point links at the hosted Sphinx manual to the matching local page."""
    def local(match):
        name = (match.group(1) or "index.html")[:-5]
        if not (match.group(1) or ".html").endswith(".html") or name not in pages:
            return None, None
        relative = posixpath.relpath(name + ".md", posixpath.dirname(page) or ".")
        return name, relative + (match.group(2) or "")

    def autolink(match):
        name, target = local(match)
        if target is None:
            return match.group(0)
        return "[" + (page_title(pages[name]) or name) + "](" + target + ")"

    def link(match):
        target = local(match)[1]
        return match.group(0) if target is None else "](" + target + ")"

    return RTD_TARGET.sub(link, RTD_AUTOLINK.sub(autolink, text))


def rst_to_markdown(rst, page):
    markdown = pypandoc.convert_text(
        strip_toctrees(rst), "gfm", format="rst", extra_args=["--wrap=none"])
    markdown = FENCE_LANGUAGE.sub(r"\1\2", markdown)
    lines = convert_alerts(markdown.splitlines())
    for index, line in outside_fences(lines, skip_tables=True):
        line = IMAGE.sub(convert_image, line)
        line = DIV.sub(r'<div\1 markdown="1">', line)
        line = PAREN_LIST.sub(r"\1.\2", line)
        # Python-Markdown has no backslash line break
        if line.endswith("\\"):
            line = line[:-1] + "  "
        lines[index] = line
    lines = join_image_rows(lines)
    title = posixpath.basename(page).replace("_", " ").title()
    return "\n".join(fix_headings(lines, title)) + "\n"


def page_title(markdown):
    for _, line in outside_fences(markdown.splitlines()):
        match = HEADING.match(line)
        if match and len(match.group(1)) == 1:
            return match.group(2).strip()
    return None


def dive_index_rst(ref):
    return "\n".join([
        dive_docs.README_INCLUDE,
        "",
        "The pages below are vendored from the `DIVE manual`_ at the revision VIAME",
        "currently ships (``" + ref[:12] + "``). The upstream site is authoritative for",
        "anything newer.",
        "",
        ".. _DIVE manual: " + dive_docs.DIVE_SITE,
        "",
    ])


def dive_page(text):
    lines = text.splitlines()
    for index, line in outside_fences(lines):
        lines[index] = dive_docs.MD_LINK.sub(dive_docs.rewrite_link, line)
    return "\n".join(lines) + "\n"


def collect_dive(sources, pages, assets):
    """DIVE writes for mkdocs-material already, so its pages are used as they are."""
    ref = dive_docs.dive_ref()
    source = dive_docs.source_docs(ref)
    sources[DIVE_INDEX] = (HERE / DIVE_DIR / "index.rst", dive_index_rst(ref))
    if source is None:
        log.warning("DIVE manual unavailable, building without it")
        return []
    nav = []
    for caption, group in dive_docs.PAGE_GROUPS:
        entries = []
        for name, title in group:
            origin = source / name
            if not origin.is_file():
                log.warning("skipping missing DIVE page %s", name)
                continue
            text = origin.read_text(encoding="utf-8")
            pages[DIVE_DIR + "/" + name[:-3]] = dive_page(text)
            for asset in set(dive_docs.ASSET.findall(text)):
                if (source / asset).is_file():
                    assets[DIVE_DIR + "/" + asset] = source / asset
            entries.append({title: DIVE_DIR + "/" + name})
        nav.append({caption: entries})
    return nav


def build_nav(name, raw, pages, dive_nav):
    entries = []
    base = posixpath.dirname(name)
    for title, target in toctree_entries(raw.get(name, "")):
        if re.match(r"https?://", target):
            match = RTD_LINK.fullmatch(target)
            local = (match.group(1) or "index.html")[:-5] if match else None
            entries.append({title: local + ".md" if local in pages else target})
            continue
        child = posixpath.normpath(posixpath.join(base, target))
        if child not in pages:
            log.warning("toctree entry %s in %s has no page", target, name)
            continue
        children = dive_nav if child == DIVE_INDEX else build_nav(
            child, raw, pages, dive_nav)
        label = title or page_title(pages[child]) or child
        if children:
            # Material only folds a page into its section when it is an index
            first = child + ".md"
            if posixpath.basename(child) != "index":
                first = {"Overview": first}
            entries.append({label: [first] + children})
        else:
            entries.append({label: child + ".md"} if title else child + ".md")
    return entries


def generate():
    sources, pages, assets = {}, {}, {}
    sources["index"] = (HERE / "index.rst", None)
    for path in sorted((HERE / SECTIONS).glob("*.rst")):
        sources[SECTIONS + "/" + path.stem] = (path, None)
    dive_nav = collect_dive(sources, pages, assets)

    raw = {}
    for name, (path, text) in sources.items():
        try:
            raw[name] = flatten(path, text)
        except OSError as error:
            log.info("skipping %s: %s", name, error)
    for name, text in raw.items():
        pages[name] = rst_to_markdown(text, name)
    for name in pages:
        pages[name] = relink(pages[name], name + ".md", pages)

    for path in sorted((HERE / "_static" / "images").iterdir()):
        assets["_static/images/" + path.name] = path
    assets["favicon.ico"] = HERE.parent / "favicon.ico"

    nav = [{"Home": "index.md"}] + [
        entry for entry in build_nav("index", raw, pages, dive_nav)
        if entry != {"Documentation Overview": "index.md"}]
    return pages, assets, nav


def on_config(config):
    pages, assets, nav = generate()
    _generated["pages"], _generated["assets"] = pages, assets
    config["nav"] = nav
    return config


def on_files(files, config):
    for name, markdown in _generated["pages"].items():
        files.append(File.generated(config, name + ".md", content=markdown))
    for name, path in _generated["assets"].items():
        files.append(File.generated(config, name, abs_src_path=str(path)))
    return files
