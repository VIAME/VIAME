"""Add the pages that live outside the MkDocs directory to the VIAME manual.

The example READMEs and the DIVE manual are handed to MkDocs as generated files
laid out like the Sphinx build, so nothing is copied into the source tree. Each
Sphinx wrapper page names the README it shows; its Markdown twin is used here.
"""

import logging
import posixpath
import re
import sys
from pathlib import Path

from mkdocs.structure.files import File

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dive_docs  # noqa: E402

log = logging.getLogger("mkdocs.hooks.viame")

SECTIONS = "sections"
DIVE_DIR = SECTIONS + "/dive"

INCLUDE = re.compile(r"^\.\. include:: *(\S+) *\n((?:[ \t]+:[\w-]+:.*\n)*)", re.M)
RTD = r"https?://viame\.readthedocs\.io/en/latest/?([^\s)#\"'>]*)(#[^\s)\"'>]*)?"
RTD_AUTOLINK = re.compile("<" + RTD + ">")
RTD_TARGET = re.compile(r"\]\(" + RTD + r"\)")
RTD_HREF = re.compile('href="' + RTD + '"')
FENCE = re.compile(r"^\s*(```+|~~~+)")
HEADING = re.compile(r"^(#+) +(.*)$")
ALERT = re.compile(r"^> \[!(\w+)\] *$")
REPO_IMAGE = re.compile(r"(?:\.\./)+docs/manual/(_static/images/)")

# Sections the Sphinx manual lists in its sidebar, given pages of their own
SPLITS = {
    "sections/search_and_rapid_model_generation": {
        "Rapid Model Generation": "sections/rapid_model_generation"},
}


def included(text, directory):
    """Markdown twin of what the include directives of a Sphinx page pull in."""
    parts = []
    for match in INCLUDE.finditer(text):
        options = dict(re.findall(r":([\w-]+): *(.*)", match.group(2)))
        target = (directory / match.group(1)).resolve().with_suffix(".md")
        body = target.read_text(encoding="utf-8")
        for key, value in options.items():
            if value.startswith(".. "):
                options[key] = "<!-- " + value[3:] + " -->"
        if "start-after" in options:
            body = body.partition(options["start-after"])[2]
        if "end-before" in options:
            body = body.partition(options["end-before"])[0]
        parts.append(body.strip("\n"))
    return "\n\n".join(parts) + "\n"


def outside_fences(lines):
    in_fence = False
    for index, line in enumerate(lines):
        if FENCE.match(line):
            in_fence = not in_fence
        elif not in_fence:
            yield index, line


def page_title(markdown):
    for _, line in outside_fences(markdown.splitlines()):
        match = HEADING.match(line)
        if match and len(match.group(1)) == 1:
            return match.group(2).strip()
    return None


def convert_alerts(lines):
    """GitHub alerts, which the READMEs use, become Material admonitions."""
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


def adapt(markdown):
    return "\n".join(convert_alerts(markdown.splitlines())) + "\n"


def slug(title):
    return "#" + re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def split_section(markdown, title):
    """Return the page without the titled section, and the section as a page."""
    lines = markdown.splitlines()
    start = end = None
    for index, line in outside_fences(lines):
        match = HEADING.match(line)
        if not match:
            continue
        if start is None and match.group(2).strip() == title:
            start, level = index, len(match.group(1))
        elif start is not None and len(match.group(1)) <= level:
            end = index
            break
    if start is None:
        return markdown, None
    section = lines[start:end]
    for index, line in outside_fences(section):
        if HEADING.match(line):
            section[index] = line[level - 1:]
    rest = lines[:start] + (lines[end:] if end is not None else [])
    return "\n".join(rest).rstrip("\n") + "\n", "\n".join(section).rstrip("\n") + "\n"


def relink(text, page, pages):
    """Point links at the hosted Sphinx manual to the matching local page."""
    def local(match, suffix):
        name = (match.group(1) or "index.html")[:-5]
        if not (match.group(1) or ".html").endswith(".html") or name not in pages:
            return None, None
        anchor = match.group(2) or ""
        for title, target in SPLITS.get(name, {}).items():
            if anchor == slug(title):
                name, anchor = target, ""
        relative = posixpath.relpath(name + suffix, posixpath.dirname(page) or ".")
        return name, relative + anchor

    def autolink(match):
        name, target = local(match, ".md")
        if target is None:
            return match.group(0)
        title = page_title(pages[name].content_string)
        return "[" + (title or name) + "](" + target + ")"

    def link(match):
        target = local(match, ".md")[1]
        return match.group(0) if target is None else "](" + target + ")"

    def href(match):
        target = local(match, ".html")[1]
        return match.group(0) if target is None else 'href="' + target + '"'

    text = RTD_TARGET.sub(link, RTD_AUTOLINK.sub(autolink, text))
    return RTD_HREF.sub(href, text)


def promote_headings(markdown):
    """Raise every heading a level when a page part starts below the top one."""
    lines = markdown.splitlines()
    body = list(outside_fences(lines))
    for position, (index, line) in enumerate(body):
        following = body[position + 1][1] if position + 1 < len(body) else ""
        if line.startswith("# ") or (line.strip() and re.fullmatch(r"=+", following)):
            return markdown
    for index, line in body:
        if line.startswith("##"):
            lines[index] = line[1:]
    return "\n".join(lines) + "\n"


def dive_index(ref):
    readme = included(dive_docs.README_INCLUDE + "\n", HERE / DIVE_DIR)
    return readme + "\n".join([
        "",
        "The pages below are vendored from the [DIVE manual](" + dive_docs.DIVE_SITE +
        ") at the revision VIAME currently ships (`" + ref[:12] + "`). The upstream "
        "site is authoritative for anything newer.",
        "",
    ])


def dive_page(text):
    lines = text.splitlines()
    for index, line in outside_fences(lines):
        lines[index] = dive_docs.MD_LINK.sub(dive_docs.rewrite_link, line)
    return "\n".join(lines) + "\n"


def collect_dive(pages, assets):
    """DIVE writes for mkdocs-material already, so its pages are used as they are."""
    ref = dive_docs.dive_ref()
    source = dive_docs.source_docs(ref)
    pages[DIVE_DIR + "/index"] = dive_index(ref)
    if source is None:
        log.warning("DIVE manual unavailable, building without it")
        return
    for name in dive_docs.PAGES:
        origin = source / name
        if not origin.is_file():
            log.warning("skipping missing DIVE page %s", name)
            continue
        text = origin.read_text(encoding="utf-8")
        pages[DIVE_DIR + "/" + name[:-3]] = dive_page(text)
        for asset in set(dive_docs.ASSET.findall(text)):
            if (source / asset).is_file():
                assets[DIVE_DIR + "/" + asset] = source / asset


def generate():
    pages, assets = {}, {}
    for path in sorted((HERE / SECTIONS).glob("*.rst")):
        try:
            text = included(path.read_text(encoding="utf-8"), path.parent)
        except OSError as error:
            log.info("skipping %s: %s", path.name, error)
            continue
        if text.strip():
            pages[SECTIONS + "/" + path.stem] = text
    for name, sections in SPLITS.items():
        for title, target in sections.items():
            pages[name], section = split_section(pages[name], title)
            if section:
                pages[target] = section
                pages[name] = pages[name].replace(
                    "](" + slug(title) + ")",
                    "](" + posixpath.basename(target) + ".md)")
    collect_dive(pages, assets)
    for name in pages:
        pages[name] = promote_headings(pages[name])
        # READMEs address images from their own folder
        pages[name] = REPO_IMAGE.sub("../" * name.count("/") + r"\1", pages[name])

    for path in sorted((HERE / "_static" / "images").iterdir()):
        assets["_static/images/" + path.name] = path
    assets["favicon.ico"] = HERE.parent / "favicon.ico"
    return pages, assets


def on_files(files, config):
    pages, assets = generate()
    for name, markdown in pages.items():
        files.append(File.generated(config, name + ".md", content=markdown))
    for name, path in assets.items():
        files.append(File.generated(config, name, abs_src_path=str(path)))
    return files


def on_page_markdown(markdown, page, config, files):
    name = page.file.src_uri
    if not name.startswith(DIVE_DIR + "/") or name == DIVE_DIR + "/index.md":
        markdown = adapt(markdown)
    pages = {f.src_uri[:-3]: f for f in files.documentation_pages()}
    return relink(markdown, name, pages)
