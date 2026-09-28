#!/usr/bin/env python
"""Publish a built VIAME package to its permanent download locations.

Both mirrors keep ONE file per platform/flavour whose contents are replaced on
every release, so the download links in README.md never change:

  Girder    a new version of the same file inside the same item; the item's
            /api/v1/item/<id>/download URL is stable across versions.
  Drive     files.update on the same file id (rclone does this when it
            overwrites a same-named file), so /file/d/<id>/view is stable.

Only the version text in README.md changes from release to release, and the
first run also writes the URLs it just created.

Credentials come from the environment, never from the repository:

  GIRDER_API_KEY   data.kitware.com key with write access to the folder
  RCLONE_EXE       path to rclone.exe        (default: rclone on PATH)
  RCLONE_REMOTE    configured remote name    (default: viame-drive)

Usage:
  python publish_release.py <zip> --platform windows --flavor gpu [--readme R]
  python publish_release.py <zip> ... --skip-drive --dry-run
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

import requests

GIRDER_API = "https://data.kitware.com/api/v1"
GIRDER_FOLDER_ID = "58d4c1178d777f0aef5d8920"
DRIVE_FOLDER_ID = "0AO5BaXFCsZRXUk9PVA"

# One permanent name per platform/flavour; contents are replaced each release.
RELEASE_NAMES = {
    ("windows", "gpu"): "VIAME-Windows-64Bit-GPU-latest.zip",
    ("windows", "cpu"): "VIAME-Windows-64Bit-CPU-latest.zip",
    ("linux", "gpu"): "VIAME-Linux-64Bit-GPU-latest.tar.gz",
    ("linux", "cpu"): "VIAME-Linux-64Bit-CPU-latest.tar.gz",
}

# The README rows this script owns, keyed the same way. The label is matched
# verbatim inside the link text, so a row is found regardless of its version.
README_LABELS = {
    ("windows", "gpu"): ("Windows, GPU Enabled", ".zip"),
    ("windows", "cpu"): ("Windows, CPU Only", ".zip"),
    ("linux", "gpu"): ("Linux, GPU Enabled", ".tar.gz"),
    ("linux", "cpu"): ("Linux, CPU Only", ".tar.gz"),
}

CHUNK_SIZE = 128 * 1024 * 1024
RETRIES = 4


def log(message):
    print("[publish] " + message, flush=True)


def sha256_and_size(path):
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


# ---------------------------------------------------------------------------
# Girder
# ---------------------------------------------------------------------------
class Girder(object):
    def __init__(self, api_key):
        self.session = requests.Session()
        response = self.session.post(GIRDER_API + "/api_key/token",
                                     params={"key": api_key, "duration": 1})
        response.raise_for_status()
        token = response.json()["authToken"]["token"]
        self.session.headers.update({"Girder-Token": token})

    def get(self, path, **params):
        response = self.session.get(GIRDER_API + path, params=params)
        response.raise_for_status()
        return response.json()

    def post(self, path, **params):
        response = self.session.post(GIRDER_API + path, params=params)
        response.raise_for_status()
        return response.json()

    def find_item(self, folder_id, name):
        matches = self.get("/item", folderId=folder_id, name=name)
        return matches[0] if matches else None

    def upload_chunks(self, upload_id, path, size):
        """Feed the file to an open upload, retrying a failed chunk."""
        sent = 0
        with open(path, "rb") as handle:
            while sent < size:
                block = handle.read(CHUNK_SIZE)
                for attempt in range(RETRIES):
                    try:
                        response = self.session.post(
                            GIRDER_API + "/file/chunk",
                            params={"uploadId": upload_id, "offset": sent},
                            data=block)
                        response.raise_for_status()
                        break
                    except requests.RequestException as error:
                        if attempt == RETRIES - 1:
                            raise
                        log("  chunk at %d failed (%s); retrying" % (sent, error))
                sent += len(block)
                log("  uploaded %.1f / %.1f GB" % (sent / 1e9, size / 1e9))
        return response.json()

    def publish(self, path, name, size):
        """Replace the contents of the permanent file, or create it once."""
        item = self.find_item(GIRDER_FOLDER_ID, name)
        if item is None:
            log("Creating Girder item %s" % name)
            item = self.post("/item", folderId=GIRDER_FOLDER_ID, name=name)
            upload = self.post("/file", parentType="item", parentId=item["_id"],
                               name=name, size=size)
        else:
            files = self.get("/item/%s/files" % item["_id"])
            if not files:
                upload = self.post("/file", parentType="item",
                                   parentId=item["_id"], name=name, size=size)
            else:
                log("Replacing contents of Girder file %s" % files[0]["_id"])
                upload = self.post("/file/%s/contents" % files[0]["_id"], size=size)
        uploaded = self.upload_chunks(upload["_id"], path, size)
        return item["_id"], uploaded


def girder_verify(girder, item_id, size):
    files = girder.get("/item/%s/files" % item_id)
    if not files:
        raise RuntimeError("Girder item %s has no file after upload" % item_id)
    if files[0]["size"] != size:
        raise RuntimeError("Girder size mismatch: %d != %d" % (files[0]["size"], size))
    return files[0]


# ---------------------------------------------------------------------------
# Google Drive (via rclone, which updates a same-named file in place)
# ---------------------------------------------------------------------------
def drive_publish(path, name, rclone, remote, dry_run=False):
    target = "%s:" % remote
    command = [rclone, "copyto", path, "%s%s" % (target, name),
               "--drive-root-folder-id", DRIVE_FOLDER_ID,
               "--drive-chunk-size", "256M",
               "--retries", str(RETRIES), "--low-level-retries", "20",
               "--stats", "30s", "--stats-one-line"]
    if dry_run:
        command.append("--dry-run")
    log("Drive: " + " ".join(command))
    subprocess.check_call(command)

    listing = subprocess.check_output(
        [rclone, "lsjson", "--hash", "%s%s" % (target, name),
         "--drive-root-folder-id", DRIVE_FOLDER_ID])
    entries = json.loads(listing.decode("utf-8"))
    return entries[0] if entries else None


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------
def update_readme(readme_path, platform, flavor, version, mirror_urls):
    """Set the version text (and, on the first run, the URLs) of the two rows
    this platform/flavour owns. Returns the lines that changed."""
    label, extension = README_LABELS[(platform, flavor)]
    with open(readme_path, "r", encoding="utf-8") as handle:
        text = handle.read()

    changed = []
    for mirror, url in sorted(mirror_urls.items()):
        if url is None:
            continue
        # * [VIAME v0.23.2 Windows, GPU Enabled, Mirror1 (.zip)](url)
        pattern = re.compile(
            r"(\* \[VIAME )v[0-9][^ ]*( %s, %s \(%s\)\]\()[^)]*(\))"
            % (re.escape(label), re.escape(mirror), re.escape(extension)))
        replacement = r"\g<1>v%s\g<2>%s\g<3>" % (version, url)
        text, count = pattern.subn(replacement, text)
        if count:
            changed.append("%s %s -> v%s" % (label, mirror, version))
        else:
            log("WARNING: no README row for '%s, %s (%s)'" % (label, mirror, extension))

    if changed:
        with open(readme_path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("package", help="the built .zip / .tar.gz to publish")
    parser.add_argument("--platform", required=True, choices=["windows", "linux"])
    parser.add_argument("--flavor", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--version", help="release version (default: from RELEASE_NOTES.md)")
    parser.add_argument("--readme", help="README.md to update (skipped when omitted)")
    parser.add_argument("--skip-girder", action="store_true")
    parser.add_argument("--skip-drive", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="hash and resolve targets, upload nothing")
    args = parser.parse_args()

    if not os.path.isfile(args.package):
        raise SystemExit("No such package: %s" % args.package)

    version = args.version
    if not version:
        source_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        with open(os.path.join(source_dir, "RELEASE_NOTES.md"), encoding="utf-8") as handle:
            version = handle.readline().split()[0]
    version = version.lstrip("v")

    name = RELEASE_NAMES[(args.platform, args.flavor)]
    log("Publishing %s as %s (v%s)" % (args.package, name, version))

    digest, size = sha256_and_size(args.package)
    log("sha256 %s  (%.2f GB)" % (digest, size / 1e9))
    sums_path = args.package + ".sha256"
    with open(sums_path, "w", encoding="utf-8") as handle:
        handle.write("%s  %s\n" % (digest, name))

    mirror_urls = {}

    if not args.skip_girder:
        api_key = os.environ.get("GIRDER_API_KEY")
        if not api_key:
            raise SystemExit("GIRDER_API_KEY is not set")
        girder = Girder(api_key)
        if args.dry_run:
            existing = girder.find_item(GIRDER_FOLDER_ID, name)
            log("Girder item: %s" % (existing["_id"] if existing else "(would create)"))
            if existing:
                mirror_urls["Mirror2"] = "%s/item/%s/download" % (GIRDER_API, existing["_id"])
        else:
            item_id, _ = girder.publish(args.package, name, size)
            stored = girder_verify(girder, item_id, size)
            log("Girder ok: item %s, file %s" % (item_id, stored["_id"]))
            mirror_urls["Mirror2"] = "%s/item/%s/download" % (GIRDER_API, item_id)

    if not args.skip_drive:
        rclone = os.environ.get("RCLONE_EXE", "rclone")
        remote = os.environ.get("RCLONE_REMOTE", "viame-drive")
        entry = drive_publish(args.package, name, rclone, remote, args.dry_run)
        if entry:
            log("Drive ok: id %s, size %s" % (entry.get("ID"), entry.get("Size")))
            if not args.dry_run and entry.get("Size") != size:
                raise SystemExit("Drive size mismatch: %s != %d" % (entry.get("Size"), size))
            if entry.get("ID"):
                mirror_urls["Mirror1"] = ("https://drive.google.com/file/d/%s/view?usp=sharing"
                                          % entry["ID"])

    if args.readme and mirror_urls:
        if args.dry_run:
            log("Would update README rows: %s" % ", ".join(sorted(mirror_urls)))
        else:
            for line in update_readme(args.readme, args.platform, args.flavor,
                                      version, mirror_urls):
                log("README: " + line)

    log("Done. sha256 written to %s" % sums_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
