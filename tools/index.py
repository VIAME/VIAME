#!/usr/bin/env python
# This file is part of VIAME, and is distributed under an OSI-approved
# BSD 3-Clause License. See either the root top-level LICENSE file or
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

"""
Build and manage a VIAME video search index.

The index is a folder (default "database") holding, for every indexed video
or image list ("stream"), a set of files sharing its basename:

  <name>.index             JSON manifest; marks the stream as indexed
  <name>_descriptors.csv   descriptor uids, track references, per-frame history
  <name>_tracks.csv        the object tracks the descriptors refer to
  <name>_descriptors.npy   N x D float32 descriptor matrix
  <name>_uids.txt          uid of each row
  <name>_hashes.npy        N x bits uint8 ITQ hash codes of each row
  ITQ/itq.model.*.npy      the one ITQ model shared by every stream

That file-backed layout is the default backend. The embedded PostgreSQL
backend (--backend postgres) keeps descriptors and tracks in a database
under <folder>/SQL instead, with the same ITQ files; a folder holds one or
the other, never both.

Commands:
  add      Run the ingest pipeline on videos or images and refresh the hashes
  remove   Drop streams from the index
  build    Refresh the index: convert new descriptor files, train the ITQ
           model if missing (or --retrain), hash stale streams
  list     List indexed streams
  status   Show the index (or one stream) in detail
  hash     Train an ITQ model and hash codes from an arbitrary descriptor
           source (CSV file or PostgreSQL table); the low-level tool

Examples:
  viame index add -l ingest_list.txt
  viame index add -d videos --method tracking -frate 5
  viame index add -l ingest_list.txt --method existing -id detections.csv
  viame index list
  viame index remove ingest_list
  viame index build --retrain
"""

import argparse
import json
import os
import subprocess
import sys
import time

import database
from viame.core import index_descriptors

lb1 = os.linesep

METHOD_PIPELINES = {
    "detections": "pipelines/index_generic.pipe",
    "tracking": "pipelines/index_generic.trk.pipe",
    "existing": "pipelines/index_existing.pipe",
    "frames": "pipelines/index_frame.pipe",
}

BACKENDS = ("files", "postgres")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def script_dir():
    return os.path.dirname(os.path.realpath(__file__))


def resolve_backend(args):
    backend = args.backend or index_descriptors.detect_backend(args.database)
    if backend not in BACKENDS:
        raise SystemExit("Unknown backend: %s" % backend)
    return backend


def ensure_postgres(database_dir, init=False, prompt=True):
    """Make sure the embedded server of a postgres index is running,
    initialising it when the folder has none yet (or when init is set)."""
    if init or not database.has_sql_dir(database_dir):
        ok, _ = database.init(prompt=prompt, database_dir=database_dir)
        if not ok:
            raise SystemExit("Unable to initialize the database in %s" % database_dir)
        return
    if not database.status(quiet=True, database_dir=database_dir):
        if not database.start(quiet=False, database_dir=database_dir):
            raise SystemExit("Unable to start the database in %s" % database_dir)


def model_summary(database_dir):
    itq_dir = os.path.join(database_dir, "ITQ")
    if not os.path.isdir(itq_dir):
        return None
    models = sorted(f for f in os.listdir(itq_dir)
                    if f.startswith("itq.model.") and f.endswith(".rotation.npy"))
    if not models:
        return None
    suffix = models[-1][len("itq.model."):-len(".rotation.npy")]
    try:
        digest = index_descriptors._model_hash(itq_dir, suffix)
    except OSError:
        digest = None
    return {"model": suffix, "hash": digest, "dir": itq_dir}


def file_streams(database_dir):
    """[(name, manifest or None)] for every stream with a descriptor file or array."""
    if not os.path.isdir(database_dir):
        return []
    return [(name, index_descriptors.read_bundle_manifest(database_dir, name))
            for name in index_descriptors.list_index_bundles(database_dir)]


def print_table(rows, headers):
    widths = [max(len(str(r[i])) for r in rows + [headers]) for i in range(len(headers))]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------
def cmd_add(args):
    database_dir = os.path.abspath(args.database)
    backend = resolve_backend(args)
    install = args.install or os.environ.get("VIAME_INSTALL", "")

    if args.pipeline:
        pipeline = args.pipeline
    else:
        pipeline = METHOD_PIPELINES[args.method]
    if args.method == "existing" and not args.input_detections and not args.pipeline:
        raise SystemExit("--method existing needs -id <detections.csv>")

    inputs = [(flag, value) for flag, value in
              (("-i", args.input), ("-v", args.input_video),
               ("-d", args.input_dir), ("-l", args.input_list)) if value]
    if not inputs:
        raise SystemExit("Give the media to index with -i, -v, -d or -l")

    if backend == "postgres":
        ensure_postgres(database_dir, init=args.init, prompt=not args.yes)
    os.makedirs(database_dir, exist_ok=True)

    started = time.time()
    command = [sys.executable, os.path.join(script_dir(), "run_bulk.py"),
               "--no-reset-prompt", "-p", pipeline, "-o", database_dir,
               "--index-backend", backend]
    for flag, value in inputs:
        command += [flag, value]
    if args.input_detections:
        command += ["-id", args.input_detections]
    if args.frame_rate:
        command += ["-frate", args.frame_rate]
    if install:
        command += ["-install", install]
    if args.gpu_count:
        command += ["-gpus", args.gpu_count]
    command += args.extra

    print("Ingesting with %s (%s backend)" % (os.path.basename(pipeline), backend) + lb1)
    result = subprocess.call(command)
    if result != 0:
        raise SystemExit("Ingest failed (exit code %d); see %s" % (
            result, os.path.join(database_dir, "logs")))

    if args.no_hash:
        print("Skipping hash refresh (--no-hash); run 'viame index build' later")
        return 0

    summary = refresh_index(database_dir, backend, args)
    if backend == "files":
        added = [name for name, manifest in file_streams(database_dir)
                 if manifest and manifest_time(manifest) >= started - 1]
        print(lb1 + "Indexed stream(s): " + (", ".join(added) if added else "none new")
              + "  (%d stream(s), %d descriptors in total)" % (
                  summary["bundles"], summary["descriptors"]))
    return 0


def manifest_time(manifest):
    try:
        return time.mktime(time.strptime(manifest.get("updated", ""), "%Y-%m-%dT%H:%M:%S"))
    except (ValueError, TypeError):
        return 0


def refresh_index(database_dir, backend, args):
    """Convert, train and hash for the files backend; regenerate the global
    hash table for postgres. Returns the bundle summary for files."""
    if backend == "postgres":
        if not database.build_index(backend="postgres", database_dir=database_dir):
            raise SystemExit("Unable to build the hash index")
        return {"bundles": 0, "descriptors": 0, "rehashed": 0, "trained": False}
    summary = index_descriptors.build_index_bundles(
        database_dir=database_dir,
        bit_length=args.bit_length,
        itq_iterations=args.itq_iterations,
        random_seed=args.random_seed,
        max_train_descriptors=args.max_train_descriptors,
        retrain=getattr(args, "retrain", False),
        strip_vectors=not args.keep_csv_vectors,
        verbose=not args.quiet,
    )
    return summary


# ---------------------------------------------------------------------------
# build / remove / list / status
# ---------------------------------------------------------------------------
def cmd_build(args):
    database_dir = os.path.abspath(args.database)
    if not os.path.isdir(database_dir):
        raise SystemExit("Index folder does not exist: %s" % database_dir)
    backend = resolve_backend(args)
    if backend == "postgres":
        ensure_postgres(database_dir)
    summary = refresh_index(database_dir, backend, args)
    if backend == "files":
        print("%d stream(s), %d descriptors; %d rehashed; model %s" % (
            summary["bundles"], summary["descriptors"], summary["rehashed"],
            "trained" if summary["trained"] else "reused"))
    return 0


def cmd_remove(args):
    database_dir = os.path.abspath(args.database)
    backend = resolve_backend(args)
    if backend == "postgres":
        ensure_postgres(database_dir)
        database.remove_streams(args.streams)
        for name in args.streams:
            print("Removed %s" % name)
        return 0
    missing = 0
    for name in args.streams:
        removed = index_descriptors.remove_index_bundle(database_dir, name)
        if removed:
            print("Removed %s (%d files)" % (name, len(removed)))
        else:
            print("No such stream: %s" % name)
            missing += 1
    return 1 if missing else 0


def cmd_list(args):
    database_dir = os.path.abspath(args.database)
    backend = resolve_backend(args)
    if backend == "postgres":
        ensure_postgres(database_dir)
        rows = [(name, count) for name, count in database.list_streams()]
        if not rows:
            print("No streams indexed")
            return 0
        print_table(rows, ("stream", "descriptors"))
        return 0
    streams = file_streams(database_dir)
    if not streams:
        print("No streams indexed in %s" % database_dir)
        return 0
    model = model_summary(database_dir)
    rows = []
    for name, manifest in streams:
        if manifest is None:
            rows.append((name, "-", "-", "not built"))
            continue
        state = "ready"
        if model and (manifest.get("itq_model") != model["model"]
                      or manifest.get("itq_model_hash") != model["hash"]):
            state = "stale model"
        if not os.path.exists(os.path.join(database_dir, name + index_descriptors.BUNDLE_HASHES_POSTFIX)):
            state = "no hashes"
        rows.append((name, manifest.get("count", "-"), manifest.get("updated", "-"), state))
    print_table(rows, ("stream", "descriptors", "updated", "state"))
    return 0


def cmd_status(args):
    database_dir = os.path.abspath(args.database)
    backend = resolve_backend(args)
    if args.stream:
        if backend == "postgres":
            ensure_postgres(database_dir)
            counts = dict(database.list_streams())
            if args.stream not in counts:
                raise SystemExit("No such stream: %s" % args.stream)
            print(json.dumps({"name": args.stream, "count": counts[args.stream],
                              "backend": "postgres"}, indent=2))
            return 0
        manifest = index_descriptors.read_bundle_manifest(database_dir, args.stream)
        if manifest is None:
            raise SystemExit("No manifest for stream: %s" % args.stream)
        manifest["files"] = {
            postfix: os.path.exists(os.path.join(database_dir, args.stream + postfix))
            for postfix in (index_descriptors.BUNDLE_DESCRIPTOR_CSV_POSTFIX, "_tracks.csv",
                            index_descriptors.BUNDLE_DESCRIPTOR_NPY_POSTFIX,
                            index_descriptors.BUNDLE_UIDS_POSTFIX, index_descriptors.BUNDLE_HASHES_POSTFIX)}
        print(json.dumps(manifest, indent=2))
        return 0

    info = {"folder": database_dir, "backend": backend,
            "model": model_summary(database_dir)}
    if backend == "postgres":
        info["server_running"] = database.status(quiet=True, database_dir=database_dir)
        if info["server_running"]:
            streams = database.list_streams()
            info["streams"] = len(streams)
            info["descriptors"] = sum(c for _, c in streams)
    else:
        streams = file_streams(database_dir)
        info["streams"] = len(streams)
        info["descriptors"] = sum((m or {}).get("count", 0) for _, m in streams)
    print(json.dumps(info, indent=2))
    return 0


# ---------------------------------------------------------------------------
# hash: the former generate-nn-index tool
# ---------------------------------------------------------------------------
def cmd_hash(args):
    verbose = not args.quiet
    if args.hash_only and not args.model_dir:
        raise SystemExit("--model-dir is required when using --hash-only")

    bit_length = args.bit_length
    itq_iterations = args.itq_iterations
    random_seed = args.random_seed
    normalize = args.normalize
    max_train = args.max_train_descriptors
    uuids_list_filepath = args.uuids_list

    if args.config:
        if verbose:
            print("Loading configuration from: %s" % args.config)
        config = index_descriptors.load_config(args.config)
        bit_length = config.get("bit_length", bit_length)
        itq_iterations = config.get("itq_iterations", itq_iterations)
        random_seed = config.get("random_seed", random_seed)
        normalize = config.get("normalize", normalize)
        max_train = config.get("max_descriptors", max_train)
        uuids_list_filepath = config.get("uuids_list_filepath") or uuids_list_filepath
        if config.get("source_type") != "postgres":
            raise SystemExit("Config does not specify a valid descriptor source")
        source = index_descriptors.PostgresDescriptorSource(
            host=config.get("db_host", "localhost"),
            port=config.get("db_port", 5432),
            dbname=config.get("db_name", "postgres"),
            user=config.get("db_user", "postgres"),
            password=config.get("db_pass"),
            table_name=config.get("table_name", "descriptor_index"),
            uuid_col=config.get("uuid_col", "uid"),
            element_col=config.get("element_col", "element"))
    elif args.descriptor_file:
        if args.kwiver_csv:
            source = index_descriptors.KwiverCsvDescriptorSource(args.descriptor_file)
        else:
            source = index_descriptors.CSVDescriptorSource(args.descriptor_file)
    else:
        source = index_descriptors.PostgresDescriptorSource(
            host=args.db_host, port=args.db_port, dbname=args.db_name,
            user=args.db_user, password=args.db_pass, table_name=args.table_name,
            uuid_col=args.uuid_col, element_col=args.element_col)

    train_uids = None
    if uuids_list_filepath and os.path.isfile(uuids_list_filepath):
        train_uids = index_descriptors.load_uuids_list(uuids_list_filepath)
        if verbose:
            print("Loaded %d training uids from %s" % (len(train_uids), uuids_list_filepath))

    if args.hash_only:
        index_descriptors.compute_hashes_only(
            descriptor_source=source, model_dir=args.model_dir,
            output_dir=args.output_dir, bit_length=bit_length,
            itq_iterations=itq_iterations, random_seed=random_seed,
            normalize=normalize, incremental=args.incremental, verbose=verbose)
    else:
        suffix = index_descriptors._model_suffix(bit_length, itq_iterations, random_seed)
        model_path = os.path.join(args.output_dir, "itq.model.%s.rotation.npy" % suffix)
        if os.path.exists(model_path):
            if verbose:
                print("Found existing ITQ model %s; hashing with it" % model_path)
            index_descriptors.compute_hashes_only(
                descriptor_source=source, model_dir=args.output_dir,
                output_dir=args.output_dir, bit_length=bit_length,
                itq_iterations=itq_iterations, random_seed=random_seed,
                normalize=normalize, incremental=args.incremental, verbose=verbose)
        else:
            index_descriptors.generate_nn_index(
                descriptor_source=source, output_dir=args.output_dir,
                bit_length=bit_length, itq_iterations=itq_iterations,
                random_seed=random_seed, normalize=normalize,
                pca_method=args.pca_method, init_method=args.init_method,
                max_train_descriptors=max_train, random_sample=not args.no_random_sample,
                train_uids=train_uids, report_interval=args.report_interval,
                incremental=args.incremental, verbose=verbose)
    if verbose:
        print(lb1 + "ITQ hash index complete: %s" % args.output_dir)
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def add_common(parser):
    parser.add_argument("--database", "-o", dest="database", default="database",
                        help="Index folder (default: database)")
    parser.add_argument("--backend", dest="backend", default=None, choices=BACKENDS,
                        help="Storage backend: per-video files (default for new "
                             "indexes) or embedded PostgreSQL; an existing index "
                             "keeps the backend it was built with")


def add_hash_options(parser):
    parser.add_argument("--bit-length", "-b", dest="bit_length", type=int, default=256,
                        help="Number of bits in a hash code (default: 256)")
    parser.add_argument("--itq-iterations", dest="itq_iterations", type=int, default=100,
                        help="ITQ refinement iterations (default: 100)")
    parser.add_argument("--random-seed", dest="random_seed", type=int, default=0,
                        help="Random seed for training and sampling (default: 0)")
    parser.add_argument("--max-train-descriptors", "-m", dest="max_train_descriptors",
                        type=int, default=100000,
                        help="Most descriptors sampled to train the model (default: 100000)")
    parser.add_argument("--keep-csv-vectors", dest="keep_csv_vectors", action="store_true",
                        help="Leave the raw vectors in the descriptor CSVs after "
                             "converting them (they are stripped by default)")
    parser.add_argument("--quiet", "-q", dest="quiet", action="store_true",
                        help="Suppress progress output")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="viame index",
        description="Build and manage a VIAME video search index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Commands:")[1] if "Commands:" in __doc__ else None)
    sub = parser.add_subparsers(dest="command", metavar="command")
    sub.required = True

    p = sub.add_parser("add", help="Ingest videos or images and refresh the hashes",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common(p)
    p.add_argument("-i", dest="input", default="",
                   help="Input folder, video, image, or image list (autodetect)")
    p.add_argument("-v", dest="input_video", default="", help="A single video")
    p.add_argument("-d", dest="input_dir", default="",
                   help="A folder of videos or image folders")
    p.add_argument("-l", dest="input_list", default="", help="A text list of image files")
    p.add_argument("--method", dest="method", default="detections",
                   choices=sorted(METHOD_PIPELINES),
                   help="What to index: generic detections, detection and tracking, "
                        "existing detections (-id), or whole frames")
    p.add_argument("-p", dest="pipeline", default="",
                   help="Custom index pipeline (overrides --method)")
    p.add_argument("-id", dest="input_detections", default="",
                   help="Detections to index around (--method existing)")
    p.add_argument("-frate", dest="frame_rate", default="",
                   help="Processing frame rate for videos")
    p.add_argument("-gpus", dest="gpu_count", default="", help="GPUs to use")
    p.add_argument("-install", dest="install", default="",
                   help="VIAME install folder (default: $VIAME_INSTALL)")
    p.add_argument("--init", dest="init", action="store_true",
                   help="PostgreSQL: reset the database before ingesting")
    p.add_argument("--yes", "-y", dest="yes", action="store_true",
                   help="Do not prompt before resetting a database")
    p.add_argument("--no-hash", dest="no_hash", action="store_true",
                   help="Ingest only; do not refresh the hashes")
    add_hash_options(p)
    p.add_argument("extra", nargs=argparse.REMAINDER,
                   help="Further options passed to the batch runner (after --)")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("remove", help="Drop streams from the index")
    add_common(p)
    p.add_argument("streams", nargs="+", metavar="stream", help="Stream basename(s)")
    p.set_defaults(func=cmd_remove)

    p = sub.add_parser("build", help="Refresh the index: convert, train, hash",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common(p)
    p.add_argument("--retrain", dest="retrain", action="store_true",
                   help="Retrain the ITQ model and rehash every stream")
    add_hash_options(p)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("list", help="List indexed streams")
    add_common(p)
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("status", help="Show the index, or one stream, in detail")
    add_common(p)
    p.add_argument("stream", nargs="?", default="", help="Stream basename")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("hash", help="Train an ITQ model and hash codes from a "
                                    "descriptor source (the low-level tool)",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--config", "-c", dest="config", help="JSON config file")
    src.add_argument("--descriptor-file", "-f", dest="descriptor_file",
                     help="CSV of descriptors (uid,val1,val2,...), or a kwiver "
                          "track-descriptor CSV with --kwiver-csv")
    p.add_argument("--kwiver-csv", dest="kwiver_csv", action="store_true",
                   help="Read --descriptor-file as a kwiver track-descriptor CSV")
    p.add_argument("--output-dir", "-o", dest="output_dir", default="database/ITQ",
                   help="Where the model and hash files go")
    p.add_argument("--db-host", default="localhost")
    p.add_argument("--db-port", type=int, default=5432)
    p.add_argument("--db-name", default="postgres")
    p.add_argument("--db-user", default="postgres")
    p.add_argument("--db-pass", default=None)
    p.add_argument("--table-name", default="DESCRIPTOR")
    p.add_argument("--uuid-col", default="UID")
    p.add_argument("--element-col", default="VECTOR_DATA")
    p.add_argument("--bit-length", "-b", dest="bit_length", type=int, default=256)
    p.add_argument("--itq-iterations", "-i", dest="itq_iterations", type=int, default=100)
    p.add_argument("--random-seed", "-r", dest="random_seed", type=int, default=0)
    p.add_argument("--normalize", dest="normalize", type=int, default=None,
                   help="Normalization order for input vectors (default: none)")
    p.add_argument("--max-train-descriptors", "-m", dest="max_train_descriptors",
                   type=int, default=100000)
    p.add_argument("--uuids-list", dest="uuids_list",
                   help="File of uids to train on (one per line)")
    p.add_argument("--no-random-sample", dest="no_random_sample", action="store_true")
    p.add_argument("--pca-method", dest="pca_method", default="cov_eig",
                   choices=["cov_eig", "direct_svd"])
    p.add_argument("--init-method", dest="init_method", default="svd", choices=["svd", "qr"])
    p.add_argument("--hash-only", dest="hash_only", action="store_true",
                   help="Only compute hashes with an existing model (--model-dir)")
    p.add_argument("--model-dir", dest="model_dir", help="Folder holding an existing model")
    p.add_argument("--incremental", dest="incremental", action="store_true",
                   help="Only hash descriptors not already in the hash store")
    p.add_argument("--report-interval", dest="report_interval", type=float, default=1.0)
    p.add_argument("--quiet", "-q", dest="quiet", action="store_true")
    p.set_defaults(func=cmd_hash)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "add":
        # argparse.REMAINDER keeps a leading "--"
        args.extra = [a for a in args.extra if a != "--"]
    try:
        return args.func(args)
    except SystemExit:
        raise
    except Exception as e:
        print("Error: %s" % e, file=sys.stderr)
        if not getattr(args, "quiet", False):
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
