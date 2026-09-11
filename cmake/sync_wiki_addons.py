#!/usr/bin/env python
# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Sync the Model Zoo wiki page's download links to download_viame_addons.csv.

The CSV is the source of truth: the build fetches from it, so a wiki link that
disagrees hands users a package the build will never install. Nothing in either
file references the other, which is how they drift.

WIKI_SECTIONS below is that missing link, and it is deliberately explicit rather
than inferred. Matching a CSV row to a wiki section by URL only works while the
two already agree, which is precisely when no sync is needed.

Versions are read back from the server: a Girder item named
``VIAME-Default-Fish-Models-v2.3.zip`` sets that section's **Version:** to v2.3.
Links that are not Girder items (Google Drive, data.kitware.com) have their URL
synced but their version left alone, since there is no name to read.

  cmake/sync_wiki_addons.py                 # report drift, change nothing
  cmake/sync_wiki_addons.py --apply         # edit the wiki clone
  cmake/sync_wiki_addons.py --apply --push  # edit, commit and push

Exit status is 1 when drift is found in the default reporting mode, so this can
gate CI.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request

WIKI_REMOTE = 'git@github.com:VIAME/VIAME.wiki.git'
WIKI_PAGE = 'Model-Zoo-and-Add-Ons.md'
GIRDER_ITEM_API = 'https://viame.kitware.com/api/v1/item/%s'

# CSV key -> wiki section heading (matched by prefix, so headings can carry
# trailing words). Only entries listed here are synced.
#
# Deliberately absent:
#   ARCTIC-SEAL   the wiki links a different artifact (a data.kitware.com
#                 Windows/Linux pair), not the package the CSV carries
#   ALIGN-CAMERAS no section exists on the page yet
WIKI_SECTIONS = {
    'DEFAULT-FISH':   'Fish Detector and Tracker',
    'LEARN':          'ConvNext Low-Shot Models',
    'GENERIC':        'Generic Object Proposer',
    'SAM2':           'SAM2 Auto',
    'SAM3':           'SAM3 Text Query',
    'COMMUNITY-FISH': 'Community Fish Detection',
    'SEA-LION':       'Sea Lion Models',
    'GFIT':           'Gulf Fish Identification Track',
    'HABCAM':         'HabCam Models',
    'EM-TUNA':        'EM Tuna Detectors',
    'MOUSS-DEEP7':    'MOUSS Deep 7',
    'SWFSC-PENGHEAD': 'Penguin Head',
}

# Sections whose **Version:** tracks something other than the package filename
# (GFIT's wiki version is the pipeline generation, v3.x, while its package is
# named v1.1). Their URLs sync; their version text is left alone.
VERSION_EXEMPT = {'GFIT', 'SAM3'}

DRIVE_ID_RE = re.compile(r'drive\.google\.com/file/d/([^/]+)')
DOWNLOAD_RE = re.compile(r'(\[Download[^\]]*\]\()([^)]+)(\))')
VERSION_RE = re.compile(r'(\*\*Version:\*\* *)(\S+)')
ITEM_ID_RE = re.compile(r'/item/([0-9a-f]{24})/download')
NAME_VERSION_RE = re.compile(r'-(v[0-9]+(?:\.[0-9]+)*)\.zip$', re.I)


def read_csv(path):
    """-> {key: url}, mirroring tools/add_ons.py's tolerant parsing."""
    out = {}
    with open(path, newline='') as fh:
        for row in csv.reader(fh, skipinitialspace=True):
            if len(row) < 4 or not row[0].strip() or row[0].lstrip().startswith('#'):
                continue
            out[row[0].strip()] = row[1].strip()
    return out


def remote_version(url):
    """-> 'v2.3' read from the Girder item name, or None."""
    m = ITEM_ID_RE.search(url)
    if not m:
        return None
    try:
        with urllib.request.urlopen(GIRDER_ITEM_API % m.group(1), timeout=30) as r:
            name = json.load(r).get('name', '')
    except Exception as e:
        print('  ! could not read item name for %s: %s' % (m.group(1), e))
        return None
    m = NAME_VERSION_RE.search(name)
    return m.group(1) if m else None


def same_target(a, b):
    """Google Drive share links carry a ?usp= suffix that varies by how the link
    was copied; compare by file id so that alone is not treated as drift."""
    if a == b:
        return True
    da, db = DRIVE_ID_RE.search(a), DRIVE_ID_RE.search(b)
    return bool(da and db and da.group(1) == db.group(1))


def section_bounds(text, heading):
    """-> (start, end) offsets of the section whose heading starts with `heading`."""
    for m in re.finditer(r'^### +(.*)$', text, re.M):
        if m.group(1).strip().startswith(heading):
            nxt = re.search(r'^#{2,3} ', text[m.end():], re.M)
            return m.start(), (m.end() + nxt.start()) if nxt else len(text)
    return None


def sync(text, csv_urls, check_versions=True):
    """-> (new_text, [change strings], [problem strings])"""
    changes, problems = [], []
    for key, heading in sorted(WIKI_SECTIONS.items()):
        if key not in csv_urls:
            problems.append('%s: in the wiki map but not in the CSV' % key)
            continue
        bounds = section_bounds(text, heading)
        if not bounds:
            problems.append('%s: no wiki section starting "%s"' % (key, heading))
            continue
        start, end = bounds
        seg = text[start:end]

        links = list(DOWNLOAD_RE.finditer(seg))
        if not links:
            problems.append('%s: section has no Download link' % key)
            continue
        if len(links) > 1:
            problems.append('%s: %d Download links, only the first is synced'
                            % (key, len(links)))

        want = csv_urls[key]
        if not same_target(links[0].group(2), want):
            changes.append('%-15s url  %s\n%-15s  ->  %s'
                           % (key, links[0].group(2), '', want))
            seg = seg[:links[0].start(2)] + want + seg[links[0].end(2):]

        if check_versions and key not in VERSION_EXEMPT:
            ver = remote_version(want)
            vm = VERSION_RE.search(seg)
            if ver and vm and vm.group(2) != ver:
                changes.append('%-15s ver  %s -> %s' % (key, vm.group(2), ver))
                seg = seg[:vm.start(2)] + ver + seg[vm.end(2):]

        text = text[:start] + seg + text[end:]

    for key in sorted(set(csv_urls) - set(WIKI_SECTIONS)):
        problems.append('%s: in the CSV but not mapped to a wiki section' % key)
    return text, changes, problems


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', default=os.path.join(here, 'download_viame_addons.csv'))
    ap.add_argument('--wiki', metavar='DIR',
                    help='existing wiki clone; cloned to a temp dir if omitted')
    ap.add_argument('--apply', action='store_true', help='write the changes')
    ap.add_argument('--push', action='store_true', help='commit and push (implies --apply)')
    ap.add_argument('--no-versions', action='store_true',
                    help='sync URLs only, skip the per-item name lookups')
    a = ap.parse_args()
    if a.push:
        a.apply = True

    csv_urls = read_csv(a.csv)
    print('read %d add-ons from %s' % (len(csv_urls), a.csv))

    tmp = None
    wiki = a.wiki
    if not wiki:
        tmp = tempfile.mkdtemp(prefix='viamewiki.')
        wiki = os.path.join(tmp, 'wiki')
        subprocess.check_call(['git', 'clone', '--quiet', WIKI_REMOTE, wiki])

    page = os.path.join(wiki, WIKI_PAGE)
    original = open(page).read()
    updated, changes, problems = sync(original, csv_urls, not a.no_versions)

    for p in problems:
        print('  note: %s' % p)
    if not changes:
        print('\nwiki is in sync with the CSV')
        return 0

    print('\n%d change(s):' % len(changes))
    for c in changes:
        print('  ' + c)

    if not a.apply:
        print('\nreporting only; pass --apply to write, --apply --push to publish')
        return 1

    open(page, 'w').write(updated)
    print('\nwrote %s' % page)
    if a.push:
        subprocess.check_call(['git', '-C', wiki, 'add', WIKI_PAGE])
        subprocess.check_call(['git', '-C', wiki, 'commit', '-q', '-m',
                               'Sync model download links with download_viame_addons.csv'])
        subprocess.check_call(['git', '-C', wiki, 'push', '--quiet', 'origin', 'master'])
        print('pushed to %s' % WIKI_REMOTE)
    return 0


if __name__ == '__main__':
    sys.exit(main())
