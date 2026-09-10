#!/usr/bin/env python3
"""Which registered algorithm names are backed by OpenCV.

Reads the preprocessed register_algorithms.cxx from expand_registry.sh, so
the registration macros are already expanded and every `add_factory` names a
literal implementation class. A name is OpenCV-backed when the file defining
that class -- or its header/source twin -- includes `opencv2/`.

Run from the repository root:

    design/scripts/expand_registry.sh <build dir> /tmp/pp
    viame registry-dump --json --output /tmp/registry.json
    python3 design/scripts/ocv_backed.py /tmp/pp /tmp/registry.json
"""
import re, os, glob, subprocess, sys, collections

# Where expand_registry.sh put the preprocessed registration units, and the
# registry to cross-check against.
PP = sys.argv[1] if len(sys.argv) > 1 else "pp"

ocv_files = set(subprocess.check_output(
    ["git", "grep", "-ln", "opencv2/", "--", "library", "tools", "plugins"],
    text=True).split())

def twins(path):
    stem = path.rsplit('.', 1)[0]
    return [p for p in (stem + '.h', stem + '.cxx', stem + '.txx')
            if os.path.exists(p)]

def touches_ocv(path):
    return path in ocv_files or any(t in ocv_files for t in twins(path))

# class name -> (name it registers under, file that defines it), from the
# PLUGGABLE_IMPL macros. The plain form takes the class name; the _NAMED form
# gives the name explicitly. `plugin_name()` is a static function, so the
# preprocessor leaves the call standing and this is what resolves it.
impl_pat = re.compile(
    r'PLUGGABLE_IMPL(?:_BASIC)?_NAMED\s*\(\s*(\w+)\s*,\s*"([^"]+)"'
    r'|PLUGGABLE_IMPL(?:_BASIC)?\s*\(\s*(\w+)\s*,'
    r'|PLUGGABLE_CONSTRUCTOR\s*\(\s*(\w+)\s*,')

# A handful of classes spell plugin_name() out rather than taking it from a
# macro. Those are the ones the macro scan cannot see.
spelled_pat = re.compile(
    r'class\s+\w*\s*(\w+)\s*$|'
    r'static\s+std::string\s+plugin_name\s*\(\s*\)\s*\{\s*return\s+"([^"]+)"')
defining = {}
plugin_name = {}
for path in subprocess.check_output(
        ["git", "ls-files", "library", "plugins", "tools"], text=True).split():
    if not path.endswith(('.h', '.cxx', '.txx')):
        continue
    text = open(path, errors='replace').read()
    current = None
    for m in impl_pat.finditer(text):
        cls = m.group(1) or m.group(3) or m.group(4)
        defining.setdefault(cls, path)
        if m.group(2):
            plugin_name[cls] = m.group(2)
        else:
            plugin_name.setdefault(cls, cls)

    # `static std::string plugin_name() { return "x"; }` written out inside a
    # class: attribute it to the nearest preceding class declaration.
    for m in re.finditer(
            r'\bclass\s+(?:[A-Z][A-Z0-9_]*\s+)?(\w+)\s*(?::|\{)|'
            r'static\s+std::string\s+plugin_name\s*\(\s*\)\s*'
            r'\{\s*return\s+"([^"]+)"', text):
        if m.group(1):
            current = m.group(1)
        elif m.group(2) and current:
            plugin_name[current] = m.group(2)
            defining.setdefault(current, path)

fact_pat = re.compile(
    r'add_factory\s*<\s*([\w:]+)\s*,\s*([\w:]+)\s*>\s*\(\s*'
    r'(?:"([^"]+)"|([\w:]+)::plugin_name\s*\(\s*\))\s*\)')

rows = []
for i in sorted(glob.glob(os.path.join(PP, "*.i"))):
    plugin = os.path.basename(i)[:-2]
    text = open(i, errors='replace').read()
    for m in fact_pat.finditer(text):
        iface = m.group(1).split('::')[-1]
        cls = m.group(2).split('::')[-1]
        if m.group(3) is not None:
            name = m.group(3)
        else:
            name = plugin_name.get(m.group(4).split('::')[-1])
        src = defining.get(cls)
        rows.append((name or "?" + cls, iface, cls, plugin, src,
                     bool(src and touches_ocv(src))))

seen = set()
uniq = []
for r in rows:
    key = (r[0], r[1], r[2])
    if key in seen:
        continue
    seen.add(key)
    uniq.append(r)

ocv = [r for r in uniq if r[5]]
print("%d registered (name, interface, class) triples; %d OpenCV-backed\n"
      % (len(uniq), len(ocv)))
print("%-30s %-30s %-26s %s" % ("NAME", "INTERFACE", "CLASS", "DEFINED IN"))
for name, iface, cls, plugin, src, _ in sorted(ocv):
    print("%-30s %-30s %-26s %s" % (name, iface, cls, src))

# Cross-check against what the install actually registers: a (name,
# interface) this scan produces that is not there is a parse error.
# `viame registry-dump --json --output <path>` writes it.
import json
registry = sys.argv[2] if len(sys.argv) > 2 else "tests/baseline/registry.json"
recorded = json.load(open(registry))["algorithms"]
wrong = [(n, i) for n, i, _, _, _, _ in uniq
         if n not in recorded.get(i, {})]
if wrong:
    print("\nNOT IN %s (parse error):" % registry)
    for n, i in sorted(wrong):
        print("  %-28s %s" % (n, i))

missing = [r for r in uniq if r[4] is None]
if missing:
    print("\nno PLUGGABLE_IMPL found for:")
    for name, iface, cls, plugin, _, _ in sorted(missing):
        print("  %-28s %-28s %s (%s)" % (name, iface, cls, plugin))
