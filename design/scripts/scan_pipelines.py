#!/usr/bin/env python3
"""Every config an installed pipeline asks a named implementation for.

`.pipe` and `.conf` files set an algorithm's type and then its keys under
`<block>:<type>:<key>`. This reads all of them and reports, per
implementation name, the distinct key/value sets that are actually shipped --
which is what a golden recording has to cover, because a filter that is right
on its defaults and wrong on the one option a pipeline sets is still broken.
"""
import os, re, sys, json, collections

root = sys.argv[1]
impls = set(sys.argv[2:]) or None

# `  :type   ocv_enhancer` / `relativepath foo = ...` / `block:key value`
type_re = re.compile(r'^\s*(?::|(\S+):)?type\s+(\S+)\s*$')
kv_re = re.compile(r'^\s*(\S+)\s+(.*?)\s*$')

# name -> {frozenset of (key, value)} -> [files]
found = collections.defaultdict(lambda: collections.defaultdict(list))

for dirpath, _, names in os.walk(root):
    for name in names:
        if not name.endswith(('.pipe', '.conf')):
            continue
        path = os.path.join(dirpath, name)
        text = open(path, errors='replace').read()

        # every `:type <impl>` fixes a type; the keys for it are the lines
        # whose last-but-one path element is that impl name
        types = set(m.group(2) for m in
                    (type_re.match(line) for line in text.splitlines()) if m)

        for impl in types:
            if impls and impl not in impls:
                continue
            config = {}
            for line in text.splitlines():
                line = line.split('#', 1)[0].rstrip()
                if not line or ':type' in line:
                    continue
                m = re.match(r'^\s*(?:block\s+)?(\S*[:.])?%s:([A-Za-z0-9_]+)\s+(.*)$'
                             % re.escape(impl), line)
                if m:
                    config[m.group(2)] = m.group(3).strip()
            found[impl][frozenset(config.items())].append(
                os.path.relpath(path, root))

for impl in sorted(found):
    print("### %s" % impl)
    for config, files in sorted(found[impl].items(),
                                key=lambda kv: -len(kv[1])):
        print("  %-70s  x%d  e.g. %s"
              % (json.dumps(dict(sorted(config)))[:70], len(files), files[0]))
