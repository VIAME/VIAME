#!/usr/bin/env python

# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Database management tool for VIAME.

Provides commands for initializing, starting, stopping, and indexing
a PostgreSQL database used for descriptor storage and retrieval. Every
server command takes the index folder (default "database" in the working
directory); the embedded server's data lives in its SQL subfolder.
"""

import os
import shutil
import subprocess
import sys

# Directory configuration
DATABASE_DIR = "database"
PIPELINES_DIR = "pipelines"

SQL_DIR = os.path.join(DATABASE_DIR, "SQL")
SQL_INIT_FILE = os.path.join(PIPELINES_DIR, "sql_init_table.sql")
SQL_LOG_FILE = os.path.join(DATABASE_DIR, "SQL_Log_File")


def sql_dir(database_dir=None):
    """Embedded server data directory of an index folder."""
    return os.path.join(database_dir or DATABASE_DIR, "SQL")


def sql_log_file(database_dir=None):
    return os.path.join(database_dir or DATABASE_DIR, "SQL_Log_File")


def has_sql_dir(database_dir=None):
    return os.path.isdir(sql_dir(database_dir))

# Default database schema (matching C++ processes)
DEFAULT_DB_HOST = "localhost"
DEFAULT_DB_PORT = 5432
DEFAULT_DB_NAME = "postgres"
DEFAULT_DB_USER = "postgres"
DEFAULT_TABLE_NAME = "DESCRIPTOR"
DEFAULT_CONN_STR = "postgresql:host=localhost;user=postgres"

# File-backed index: one set of files per indexed video in the database
# folder, sharing a basename (see viame.core.index_descriptors.build_index_bundles)
INDEX_POSTFIX = ".index"
DESCRIPTOR_POSTFIX = "_descriptors.csv"
TRACK_POSTFIX = "_tracks.csv"
DEFAULT_UUID_COL = "UID"
DEFAULT_ELEMENT_COL = "VECTOR_DATA"

# Global log file for command output
_log_file = ""


def _is_windows():
    return os.name == 'nt'


def _format_cmd(cmd):
    return cmd + ".exe" if _is_windows() else cmd


def _setup_log_stream():
    if not _log_file:
        return None
    if _log_file == "NULL":
        return open(os.devnull, 'w')
    log_dir = os.path.dirname(_log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return open(_log_file, 'a')


def _execute_cmd(cmd, args):
    all_args = [_format_cmd(cmd)] + args
    log = _setup_log_stream()
    try:
        subprocess.check_call(all_args, stdout=log, stderr=log)
    finally:
        if log is not None:
            log.close()


def _find_file(filename):
    if os.path.exists(filename):
        return filename
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    alt_path = os.path.join(script_dir, filename)
    if os.path.exists(alt_path):
        return alt_path
    print(f"Unable to find {filename}")
    sys.exit(1)


def _log(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()


def query_yes_no(question, default="yes"):
    """Prompt user for yes/no confirmation."""
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        prompt = " [y/n] "

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        sys.stdout.write(os.linesep)
        if default is not None and choice == '':
            return valid[default]
        if choice in valid:
            return valid[choice]
        sys.stdout.write("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


def init(log_file="", prompt=True, database_dir=None):
    """Initialize a new PostgreSQL database in the index folder."""
    global _log_file
    _log_file = log_file
    database_dir = database_dir or DATABASE_DIR

    try:
        if os.path.exists(database_dir) and prompt and not query_yes_no(
                f'\nYou are about to reset "{database_dir}", continue?'):
            return [False, True]
        # Stop any existing database first (before removing log file,
        # since pg_ctl may still hold the log file open)
        if has_sql_dir(database_dir) and not stop(quiet=True, database_dir=database_dir):
            raise RuntimeError("Could not stop the selected database; refusing to reset it")

        if log_file and os.path.exists(log_file):
            os.remove(log_file)

        # Remove existing database directory
        if os.path.exists(database_dir):
            shutil.rmtree(database_dir)
        else:
            _log("\n")

        # Initialize new database
        _log("Initializing database... ")
        _execute_cmd("initdb", ["-D", sql_dir(database_dir)])
        _execute_cmd("pg_ctl", ["-D", sql_dir(database_dir), "-w", "-t", "20",
                                "-l", sql_log_file(database_dir), "start"])
        _execute_cmd("pg_ctl", ["-D", sql_dir(database_dir), "status"])
        _execute_cmd("createuser", ["-e", "-E", "-s", "-i", "-r", "-d", "postgres"])
        _execute_cmd("psql", ["-f", _find_file(SQL_INIT_FILE), "postgres"])
        _log("Success\n")
        return [True, True]

    except Exception:
        _log("Failure\n")
        return [False, False]


def status(quiet=False, database_dir=None):
    """Check database status. Returns True if running, False otherwise."""
    global _log_file
    original = _log_file
    _log_file = "NULL" if quiet else _log_file
    try:
        _execute_cmd("pg_ctl", ["-D", sql_dir(database_dir), "status"])
        _log_file = original
        return True
    except subprocess.CalledProcessError:
        _log_file = original
        return False


def start(quiet=False, database_dir=None):
    """Start the database server."""
    global _log_file
    original = _log_file
    _log_file = "NULL" if quiet else _log_file
    try:
        _execute_cmd("pg_ctl", ["-D", sql_dir(database_dir), "-w", "-t", "20",
                                "-l", sql_log_file(database_dir), "start"])
        _log_file = original
        return True
    except Exception:
        _log_file = original
        return False


def stop(quiet=False, database_dir=None):
    """Stop only the selected server; never terminate unrelated PostgreSQL processes."""
    global _log_file
    original = _log_file
    _log_file = "NULL" if quiet else _log_file

    try:
        _execute_cmd("pg_ctl", ["-D", sql_dir(database_dir), "-m", "fast", "stop"])
        return True
    except subprocess.CalledProcessError:
        # pg_ctl status uses 3 for a server that is not running. Other
        # failures (permissions, invalid data directory) are not proof of that.
        try:
            _execute_cmd("pg_ctl", ["-D", sql_dir(database_dir), "status"])
        except subprocess.CalledProcessError as exc:
            return exc.returncode == 3
        return False
    except OSError:
        return False
    finally:
        _log_file = original


def _wait_for_port_available(port=5432, timeout=10):
    """Wait for a port to become available."""
    import socket
    import time
    start = time.time()
    while time.time() - start < timeout:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('127.0.0.1', port))
            sock.close()
            return True
        except OSError:
            sock.close()
            time.sleep(0.5)
    return False


# Removes every row belonging to one stream (video/sequence identifier).
# Child tables (keyed by UID only) go first.
REMOVE_STREAM_SQL = (
    "DELETE FROM TRACK_DESCRIPTOR_TRACK WHERE UID IN "
    "(SELECT UID FROM TRACK_DESCRIPTOR WHERE VIDEO_NAME = {stream});\n"
    "DELETE FROM TRACK_DESCRIPTOR_HISTORY WHERE UID IN "
    "(SELECT UID FROM TRACK_DESCRIPTOR WHERE VIDEO_NAME = {stream});\n"
    "DELETE FROM TRACK_DESCRIPTOR WHERE VIDEO_NAME = {stream};\n"
    "DELETE FROM DESCRIPTOR WHERE VIDEO_NAME = {stream};\n"
    "DELETE FROM OBJECT_TRACK WHERE VIDEO_NAME = {stream};"
)


def _sql_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def _psql(sql, port=None):
    """Run SQL against the local server and return its output rows."""
    cmd = [_format_cmd("psql"), "-h", DEFAULT_DB_HOST, "-p", str(port or DEFAULT_DB_PORT),
           "-d", DEFAULT_DB_NAME, "-U", DEFAULT_DB_USER, "-v", "ON_ERROR_STOP=1",
           "-tA", "-c", sql]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError("psql failed: " + result.stderr.strip())
    return [line for line in result.stdout.splitlines() if line]


def remove_streams(streams, port=None):
    """Delete every stored descriptor and track of the given streams. The
    server must be running (see start)."""
    if not streams:
        return
    _psql("\n".join(REMOVE_STREAM_SQL.format(stream=_sql_literal(s)) for s in streams), port)


def list_streams(port=None):
    """Return [(stream, descriptor count)] stored in the running server."""
    rows = _psql("SELECT VIDEO_NAME, COUNT(*) FROM DESCRIPTOR GROUP BY VIDEO_NAME "
                 "ORDER BY VIDEO_NAME", port)
    return [(name, int(count)) for name, count in (row.split("|") for row in rows)]


def build_index(log_file="", backend="files", database_dir=None):
    """
    Build the ITQ LSH index for efficient nearest neighbor search.

    With the file-backed index (backend "files", the default) every
    <name>_descriptors.csv written by an index pipeline into database_dir
    becomes a bundle: <name>_descriptors.npy, <name>_uids.txt,
    <name>_hashes.npy and a <name>.index manifest, hashed with one ITQ model
    kept in database_dir/ITQ (trained on the first build). With backend
    "postgres" the descriptors are read from the running database instead
    and a single global hash table is written, as before.
    """
    global _log_file
    _log_file = log_file

    if database_dir is None:
        database_dir = DATABASE_DIR

    if backend == "files":
        try:
            from viame.core.index_descriptors import build_index_bundles

            _log("Building file-backed ITQ index...\n")
            summary = build_index_bundles(
                database_dir=database_dir,
                bit_length=256,
                itq_iterations=100,
                random_seed=0,
                max_train_descriptors=100000,
                strip_vectors=True,
                verbose=True,
            )
            _log("  Indexed %d video(s), %d descriptors (%d rehashed)\n" % (
                summary["bundles"], summary["descriptors"], summary["rehashed"]))
            _log("Success\n")
            return True
        except Exception as e:
            _log(f"Failure: {e}\n")
            if log_file:
                _log(f"  Check log: {log_file}\n")
            import traceback
            traceback.print_exc()
            return False

    try:
        from viame.core.index_descriptors import (
            generate_nn_index,
            CSVDescriptorSource,
            PostgresDescriptorSource
        )

        _log("Building ITQ index...\n")

        # Try PostgreSQL first
        source = None
        try:
            source = PostgresDescriptorSource(
                host=DEFAULT_DB_HOST,
                port=DEFAULT_DB_PORT,
                dbname=DEFAULT_DB_NAME,
                user=DEFAULT_DB_USER,
                table_name=DEFAULT_TABLE_NAME,
                uuid_col=DEFAULT_UUID_COL,
                element_col=DEFAULT_ELEMENT_COL
            )
            source.get_all_uids()  # Test connection
            _log("  Connected to PostgreSQL database\n")
        except Exception as e:
            _log(f"  Database connection failed: {e}\n")
            # Fall back to CSV
            csv_path = os.path.join(database_dir, "descriptors.csv")
            if os.path.exists(csv_path):
                source = CSVDescriptorSource(csv_path)
                _log(f"  Using CSV file: {csv_path}\n")
            else:
                _log("  No descriptor source found (database or CSV)\n")
                return False

        output_dir = os.path.join(database_dir, "ITQ")

        generate_nn_index(
            descriptor_source=source,
            output_dir=output_dir,
            bit_length=256,
            itq_iterations=100,
            random_seed=0,
            normalize=None,
            pca_method='cov_eig',
            init_method='svd',
            max_train_descriptors=100000,
            random_sample=True,
            verbose=True
        )

        _log("Success\n")
        return True

    except Exception as e:
        _log(f"Failure: {e}\n")
        if log_file:
            _log(f"  Check log: {log_file}\n")
        import traceback
        traceback.print_exc()
        return False


def print_usage():
    print("Usage: database.py <command>")
    print("")
    print("Commands:")
    print("  init, initialize  Initialize a new database")
    print("  status            Check database status")
    print("  start             Start the database server")
    print("  stop              Stop the database server")
    print("  index [files|postgres] [dir]  Build the ITQ LSH index (file bundles by default)")
    sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()

    command = sys.argv[1].lower()

    if command in ("init", "initialize"):
        init()
    elif command == "status":
        if status():
            print("Database server is running")
        else:
            print("Database server is not running")
    elif command == "start":
        start()
    elif command == "stop":
        stop()
    elif command in ("index", "build_index"):
        # database.py index [files|postgres] [database_dir]
        backend = sys.argv[2] if len(sys.argv) > 2 else "files"
        database_dir = sys.argv[3] if len(sys.argv) > 3 else None
        if not build_index(backend=backend, database_dir=database_dir):
            sys.exit(1)
    else:
        print_usage()
