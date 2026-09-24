#!/usr/bin/env python
# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Render the README's installer download lists from download_viame_install.csv.

The CSV is the source of truth for the current releases: one line per
installer holding its file name, its MD5, then every mirror it can be fetched
from. GitHub renders the README statically, so the two "Full Desktop Binaries"
lists are regenerated from the CSV by this script and kept between the
<!-- install-links:start --> and <!-- install-links:end --> markers.

  cmake/sync_readme_installs.py            # report drift, change nothing
  cmake/sync_readme_installs.py --apply    # rewrite the README lists

Exit status is 1 when the README differs from what the CSV renders, so this
can gate CI.
"""

import argparse
import csv
import os
import re
import sys

START = '<!-- install-links:start -->'
END = '<!-- install-links:end -->'
# VIAME-v0.23.2-Windows-64Bit.zip, VIAME-CPU-v0.21.1-Linux-64Bit.tar.gz
NAME_RE = re.compile(r'^VIAME-(?P<cpu>CPU-)?(?P<version>v[0-9.]+)-(?P<platform>Windows|Linux|Mac)-'
                     r'[^.]*\.(?P<ext>zip|tar\.gz)$')


def read_csv(path):
    """-> [(name, md5, [mirror, ...])]"""
    rows = []
    with open(path, newline='') as fh:
        for row in csv.reader(fh, skipinitialspace=True):
            if not row or not row[0].strip() or row[0].lstrip().startswith('#'):
                continue
            name, md5, mirrors = row[0].strip(), row[1].strip(), [m.strip() for m in row[2:] if m.strip()]
            if not mirrors:
                raise SystemExit('%s lists no mirrors' % name)
            rows.append((name, md5, mirrors))
    return rows


def render(rows):
    """The Markdown for the two platform lists, in README order."""
    groups = {'Windows': [], 'Linux': []}
    for name, _md5, mirrors in rows:
        m = NAME_RE.match(name)
        if not m or m.group('platform') not in groups:
            raise SystemExit('cannot describe installer from its name: %s' % name)
        label = '%s %s, %s' % (m.group('version'), m.group('platform'),
                               'CPU Only' if m.group('cpu') else 'GPU Enabled')
        for i, url in enumerate(mirrors, 1):
            groups[m.group('platform')].append(
                '* [VIAME %s, Mirror%d (.%s)](%s)' % (label, i, m.group('ext'), url))
    out = []
    for platform in ('Windows', 'Linux'):
        out.append('**%s Full Desktop Binaries:** <br>' % platform)
        lines = groups[platform]
        out.extend(line + (' <br>' if i < len(lines) - 1 else '') for i, line in enumerate(lines))
        out.append('')
    return '\n'.join(out).rstrip('\n') + '\n'


def sync(text, rendered):
    start, end = text.find(START), text.find(END)
    if start < 0 or end < 0 or end < start:
        raise SystemExit('README is missing the %s / %s markers' % (START, END))
    inner_start = start + len(START)
    current = text[inner_start:end]
    wanted = '\n' + rendered
    return text[:inner_start] + wanted + text[end:], current != wanted


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--csv', default=os.path.join(here, 'download_viame_install.csv'))
    ap.add_argument('--readme', default=os.path.join(here, '..', 'README.md'))
    ap.add_argument('--apply', action='store_true', help='write the README')
    args = ap.parse_args()

    rows = read_csv(args.csv)
    with open(args.readme) as fh:
        text = fh.read()
    new_text, changed = sync(text, render(rows))
    if not changed:
        print('README installer lists match %s' % os.path.basename(args.csv))
        return 0
    if args.apply:
        with open(args.readme, 'w') as fh:
            fh.write(new_text)
        print('rewrote the installer lists in %s from %d installer(s)' % (args.readme, len(rows)))
        return 0
    print('README installer lists differ from %s; pass --apply to rewrite them' % os.path.basename(args.csv))
    return 1


if __name__ == '__main__':
    sys.exit(main())
