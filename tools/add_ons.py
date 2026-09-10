#!/usr/bin/env python
# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""List, check and install VIAME add-on model packs.

Usage:
  viame add-ons                         list add-ons, then pick ones to install
  viame add-ons list [--json]
  viame add-ons install NAME [NAME ...] [--force]
  viame add-ons install --all
  viame add-ons install NAME --from-file ARCHIVE.zip

Add-ons come from download_viame_addons.csv in the install's bin folder. Its
last column names one file, relative to configs/pipelines, that only that
add-on provides; the add-on counts as installed when the file is present.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

from pathlib import Path, PureWindowsPath

CSV_NAME = 'download_viame_addons.csv'
PIPELINES_DIR = Path('configs') / 'pipelines'

NOT_INSTALLED = 'not installed'
INSTALLED = 'installed'
UNKNOWN = 'unknown'


class Addon:
    def __init__(self, name, url, description, md5, platform, requires, marker):
        self.name = name
        self.url = url
        self.description = description
        self.md5 = md5.lower()
        self.platform = platform
        self.requires = requires
        self.marker = marker


# -----------------------------------------------------------------------------
# Locating things
# -----------------------------------------------------------------------------

def find_install(explicit=None):
    if explicit:
        return Path(explicit).resolve()
    candidates = []
    env = os.environ.get('VIAME_INSTALL')
    if env:
        candidates.append(Path(env))
    candidates.append(Path(__file__).resolve().parent.parent)
    for candidate in candidates:
        if (candidate / 'bin').is_dir() and (candidate / 'configs').is_dir():
            return candidate.resolve()
    return None


def find_csv(install, explicit=None):
    if explicit:
        return Path(explicit)
    here = Path(__file__).resolve().parent
    candidates = []
    if install:
        candidates.append(install / 'bin' / CSV_NAME)
    candidates.append(here.parent / 'cmake' / CSV_NAME)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def platform_matches(spec):
    if spec == 'LINUX-ONLY':
        return sys.platform.startswith('linux')
    if spec == 'WINDOWS-ONLY':
        return sys.platform.startswith('win')
    return True


def read_addons(csv_path):
    addons = []
    with open(csv_path, newline='') as f:
        for row in csv.reader(f, skipinitialspace=True):
            if not ''.join(row).strip():
                continue
            if len(row) < 6:
                print('warning: skipping malformed line in %s: %s'
                      % (csv_path, ','.join(row)), file=sys.stderr)
                continue
            name, url, description, md5, platform, requires = \
                [field.strip() for field in row[:6]]
            marker = row[6].strip() if len(row) > 6 else ''
            if not platform_matches(platform):
                continue
            addons.append(Addon(name, url, description, md5, platform,
                                [r.strip() for r in requires.split(',') if r.strip()],
                                marker))
    return addons


def lookup(addons, name):
    for addon in addons:
        if addon.name.lower() == name.lower():
            return addon
    return None


# -----------------------------------------------------------------------------
# Status
# -----------------------------------------------------------------------------

def marker_path(install, addon):
    return install / PIPELINES_DIR / addon.marker


def status_of(install, addon):
    if not addon.marker:
        return UNKNOWN
    return INSTALLED if marker_path(install, addon).exists() else NOT_INSTALLED


# -----------------------------------------------------------------------------
# Download and extraction
# -----------------------------------------------------------------------------

def md5_of(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest):
    if 'drive.google.com' in url:
        raise RuntimeError(
            'this add-on is hosted on Google Drive and cannot be fetched '
            'directly. Download it in a browser from\n  %s\nthen run '
            'viame add-ons install NAME --from-file <archive.zip>' % url)

    request = urllib.request.Request(url, headers={'User-Agent': 'viame-add-ons'})
    show_progress = sys.stdout.isatty()

    with urllib.request.urlopen(request) as response, open(dest, 'wb') as out:
        total = int(response.headers.get('Content-Length') or 0)
        done = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if show_progress:
                if total:
                    sys.stdout.write('\r  %3d%%  %d / %d MB' %
                                     (100 * done // total, done >> 20, total >> 20))
                else:
                    sys.stdout.write('\r  %d MB' % (done >> 20))
                sys.stdout.flush()
        if show_progress:
            sys.stdout.write('\n')


def content_prefix(names):
    """Strip single-directory wrappers, the way the build does when it
    extracts an archive, and return the prefix they form."""
    prefix = ''
    while True:
        if prefix.rstrip('/').split('/')[-2:] == ['configs', 'pipelines']:
            return prefix
        entries = set()
        for name in names:
            if not name.startswith(prefix):
                continue
            rest = name[len(prefix):]
            if not rest:
                continue
            head, sep, _ = rest.partition('/')
            entries.add(head + sep)
        if len(entries) == 1 and next(iter(entries)).endswith('/'):
            prefix += next(iter(entries))
        else:
            return prefix


def destination_for(prefix):
    parts = prefix.rstrip('/').split('/') if prefix else []
    if parts[-2:] == ['configs', 'pipelines']:
        return PIPELINES_DIR
    if parts[-1:] == ['configs']:
        return Path('configs')
    return PIPELINES_DIR


def install_archive(install, archive):
    install = Path(install).resolve()
    written = []
    with tempfile.TemporaryDirectory(prefix='.viame-addon-', dir=install) as staging:
        staging = Path(staging)
        plans = []
        with zipfile.ZipFile(archive) as zf:
            members = []
            for info in zf.infolist():
                name = info.filename.replace('\\', '/')
                if name.startswith('/') or PureWindowsPath(name).drive or '..' in name.split('/'):
                    raise ValueError('Unsafe archive path: ' + info.filename)
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Archive symlinks are not supported: ' + name)
                if not info.is_dir():
                    members.append((info, name))
            if not members:
                raise ValueError('Archive contains no files')
            prefix = content_prefix([name for _, name in members])
            dest = install / destination_for(prefix)
            seen = set()
            for i, (info, name) in enumerate(members):
                target = dest / name[len(prefix):]
                resolved = target.resolve()
                if install not in resolved.parents or resolved in seen:
                    raise ValueError('Unsafe or duplicate archive destination: ' + name)
                if target.exists() and not target.is_file():
                    raise ValueError('Archive destination is not a file: ' + str(target))
                seen.add(resolved)
                payload = staging / ('payload-%d' % i)
                with zf.open(info) as src, open(payload, 'wb') as out:
                    shutil.copyfileobj(src, out)  # verifies CRC before installation
                mode = (info.external_attr >> 16) & 0o777
                if mode:
                    os.chmod(payload, mode)
                backup = None
                if target.exists() or target.is_symlink():
                    backup = staging / ('backup-%d' % i)
                    shutil.copy2(target, backup, follow_symlinks=False)
                plans.append((target, payload, backup))
        installed = []
        created_dirs = []
        try:
            for target, payload, backup in plans:
                missing = []
                parent = target.parent
                while not parent.exists():
                    missing.append(parent)
                    parent = parent.parent
                for parent in reversed(missing):
                    parent.mkdir()
                    created_dirs.append(parent)
                os.replace(payload, target)
                installed.append((target, backup))
                written.append(target.relative_to(install).as_posix())
        except Exception:
            rollback_errors = []
            for target, backup in reversed(installed):
                try:
                    if backup is not None:
                        os.replace(backup, target)
                    else:
                        target.unlink()
                except OSError as exc:
                    rollback_errors.append(str(exc))
            for directory in reversed(created_dirs):
                try:
                    directory.rmdir()
                except OSError as exc:
                    rollback_errors.append(str(exc))
            if rollback_errors:
                # Keep backups available even when a permissions or filesystem
                # failure prevents automatic recovery.
                recovery = staging.with_name(staging.name + '-recovery')
                (staging / 'recovery.json').write_text(json.dumps([
                    {'target': str(target), 'backup': backup.name if backup else None}
                    for target, _, backup in plans], indent=2))
                staging.rename(recovery)
                raise RuntimeError('Installation rollback incomplete; backups retained at %s: %s'
                                   % (recovery, '; '.join(rollback_errors)))
            raise
    return written


def install_addon(install, addon, archive=None, force=False, ignore_checksum=False):
    temp_dir = None
    try:
        if archive is None:
            temp_dir = tempfile.mkdtemp(prefix='viame-add-ons-')
            archive = Path(temp_dir) / (addon.name + '.zip')
            print('Downloading %s' % addon.name)
            print('  from %s' % addon.url)
            download(addon.url, archive)
        else:
            archive = Path(archive)

        actual = md5_of(archive)
        if addon.md5 and actual != addon.md5:
            message = ('checksum mismatch for %s: expected %s, got %s'
                       % (addon.name, addon.md5, actual))
            if not ignore_checksum:
                raise RuntimeError(message + ' (use --ignore-checksum to install anyway)')
            print('warning: ' + message, file=sys.stderr)

        print('Installing %s into %s' % (addon.name, install))
        written = install_archive(install, archive)
        print('  %d file(s) installed' % len(written))
        if addon.marker and not marker_path(install, addon).exists():
            print('warning: %s did not provide its listed file %s; the add-on '
                  'list may be out of date' % (addon.name, addon.marker),
                  file=sys.stderr)
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)


# -----------------------------------------------------------------------------
# Listing and selection
# -----------------------------------------------------------------------------

def print_listing(install, addons, numbered=False):
    width = max([len(a.name) for a in addons] + [4])
    status_width = len(NOT_INSTALLED)
    for index, addon in enumerate(addons, 1):
        status = status_of(install, addon)
        label = ('%3d. ' % index) if numbered else '  '
        line = '%s%-*s  %-*s  %s' % (label, width, addon.name,
                                     status_width, status, addon.description)
        if addon.requires:
            line += '  [needs %s]' % ', '.join(addon.requires)
        print(line)


def listing_json(install, addons):
    return [
        {
            'name': a.name,
            'status': status_of(install, a),
            'description': a.description,
            'url': a.url,
            'md5': a.md5,
            'requires': a.requires,
            'marker': a.marker,
        }
        for a in addons
    ]


def pick_addons(addons, install):
    print()
    answer = input('Install which add-ons? (numbers or names, A for all '
                   'not yet installed, Q to quit): ').strip()
    if not answer or answer.lower() == 'q':
        return []
    if answer.lower() == 'a':
        return [a for a in addons if status_of(install, a) == NOT_INSTALLED]

    chosen = []
    for token in answer.replace(',', ' ').split():
        if token.isdigit() and 1 <= int(token) <= len(addons):
            chosen.append(addons[int(token) - 1])
            continue
        addon = lookup(addons, token)
        if addon is None:
            print('warning: ignoring unknown add-on "%s"' % token, file=sys.stderr)
            continue
        chosen.append(addon)
    return chosen


# -----------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        prog='add_ons.py',
        description='List, check and install VIAME add-on model packs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('\n\n', 1)[1])
    p.add_argument('--install-dir', metavar='DIR',
                   help='VIAME install to inspect and modify '
                        '(default: $VIAME_INSTALL or the tree this tool lives in)')
    p.add_argument('--csv', metavar='FILE',
                   help='add-on list to read (default: bin/%s in the install)' % CSV_NAME)

    sub = p.add_subparsers(dest='command', metavar='<command>')

    s = sub.add_parser('list', help='Show every add-on and whether it is installed')
    s.add_argument('--json', action='store_true', help='Print as JSON')

    s = sub.add_parser('install', help='Download and install add-ons')
    s.add_argument('names', nargs='*', metavar='NAME')
    s.add_argument('--all', action='store_true',
                   help='Install every add-on not already installed')
    s.add_argument('--from-file', metavar='ARCHIVE',
                   help='Install a single named add-on from a downloaded archive')
    s.add_argument('--force', action='store_true',
                   help='Reinstall an installed add-on')
    s.add_argument('--ignore-checksum', action='store_true',
                   help='Accept an archive whose checksum differs from the catalog')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    install = find_install(args.install_dir)
    if install is None:
        print('error: no VIAME install found; set VIAME_INSTALL or pass --install-dir',
              file=sys.stderr)
        return 1

    csv_path = find_csv(install, args.csv)
    if csv_path is None or not csv_path.is_file():
        print('error: add-on list %s not found' % (csv_path or CSV_NAME), file=sys.stderr)
        return 1

    addons = read_addons(csv_path)

    if args.command == 'list':
        if args.json:
            json.dump(listing_json(install, addons), sys.stdout, indent=2)
            print()
        else:
            print_listing(install, addons)
        return 0

    if args.command == 'install':
        if args.from_file:
            if len(args.names) != 1 or args.all:
                print('error: --from-file installs exactly one named add-on',
                      file=sys.stderr)
                return 1
            if not Path(args.from_file).is_file():
                print('error: archive not found: %s' % args.from_file, file=sys.stderr)
                return 1

        if args.all:
            chosen = [a for a in addons if status_of(install, a) == NOT_INSTALLED]
        else:
            chosen = []
            for name in args.names:
                addon = lookup(addons, name)
                if addon is None:
                    print('error: unknown add-on "%s"; known add-ons are: %s'
                          % (name, ', '.join(a.name for a in addons)), file=sys.stderr)
                    return 1
                chosen.append(addon)
        if not chosen:
            print('Nothing to install.')
            return 0
    else:
        print('VIAME install: %s' % install)
        print()
        print_listing(install, addons, numbered=True)
        if not sys.stdin.isatty():
            return 0
        chosen = pick_addons(addons, install)
        if not chosen:
            return 0
        args.force = False
        args.ignore_checksum = False
        args.from_file = None

    failures = 0
    for addon in chosen:
        status = status_of(install, addon)
        if status == INSTALLED and not args.force:
            print('%s is already installed (use --force to reinstall)' % addon.name)
            continue
        try:
            install_addon(install, addon, archive=args.from_file, force=args.force,
                          ignore_checksum=args.ignore_checksum)
        except Exception as e:
            print('error: %s: %s' % (addon.name, e), file=sys.stderr)
            failures += 1

    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
