#!/usr/bin/env python

# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Database management tool for VIAME.

Provides commands for initializing, starting, stopping, and indexing
a PostgreSQL database used for descriptor storage and retrieval.
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

# Default database schema (matching C++ processes)
DEFAULT_DB_HOST = "localhost"
DEFAULT_DB_PORT = 5432
DEFAULT_DB_NAME = "postgres"
DEFAULT_DB_USER = "postgres"
DEFAULT_TABLE_NAME = "DESCRIPTOR"
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
    subprocess.check_call(all_args, stdout=log, stderr=log)


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


def init(log_file="", prompt=True):
    """Initialize a new PostgreSQL database."""
    global _log_file
    _log_file = log_file

    if log_file and os.path.exists(log_file):
        os.remove(log_file)

    try:
        # Stop any existing database
        stop(quiet=True)

        # Remove existing database directory
        if os.path.exists(DATABASE_DIR):
            if prompt and not query_yes_no(
                    f"\nYou are about to reset \"{DATABASE_DIR}\", continue?"):
                return [False, True]
            shutil.rmtree(DATABASE_DIR)
        else:
            _log("\n")

        # Initialize new database
        _log("Initializing database... ")
        _execute_cmd("initdb", ["-D", SQL_DIR])
        _execute_cmd("pg_ctl", ["-D", SQL_DIR, "-w", "-t", "20", "-l", SQL_LOG_FILE, "start"])
        _execute_cmd("pg_ctl", ["-D", SQL_DIR, "status"])
        _execute_cmd("createuser", ["-e", "-E", "-s", "-i", "-r", "-d", "postgres"])
        _execute_cmd("psql", ["-f", _find_file(SQL_INIT_FILE), "postgres"])
        _log("Success\n")
        return [True, True]

    except Exception:
        _log("Failure\n")
        return [False, False]


def status():
    """Check database status."""
    _execute_cmd("pg_ctl", ["-D", SQL_DIR, "status"])


def start(quiet=False):
    """Start the database server."""
    global _log_file
    original = _log_file
    _log_file = "NULL" if quiet else _log_file
    try:
        _execute_cmd("pg_ctl", ["-D", SQL_DIR, "-w", "-t", "20", "-l", SQL_LOG_FILE, "start"])
        _log_file = original
        return True
    except Exception:
        _log_file = original
        return False


def stop(quiet=False):
    """Stop the database server."""
    global _log_file
    original = _log_file
    _log_file = "NULL" if quiet else _log_file

    try:
        _execute_cmd("pg_ctl", ["-D", SQL_DIR, "stop"])
    except subprocess.CalledProcessError:
        pass

    try:
        if _is_windows():
            _execute_cmd("net", ["stop", "postgresql-x64-9.5 (64-bit windows)"])
        else:
            _execute_cmd("pkill", ["postgres"])
    except subprocess.CalledProcessError:
        pass

    _log_file = original


def build_index(log_file=""):
    """
    Build ITQ LSH index for efficient nearest neighbor search.

    Uses the generate_nn_index module to create an ITQ index from descriptors
    stored in the database or a CSV file.
    """
    global _log_file
    _log_file = log_file

    try:
        from generate_nn_index import (
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
            csv_path = os.path.join(DATABASE_DIR, "descriptors.csv")
            if os.path.exists(csv_path):
                source = CSVDescriptorSource(csv_path)
                _log(f"  Using CSV file: {csv_path}\n")
            else:
                _log("  No descriptor source found (database or CSV)\n")
                return False

        output_dir = os.path.join(DATABASE_DIR, "ITQ")

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
    print("Usage: database_tool.py <command>")
    print("")
    print("Commands:")
    print("  init, initialize  Initialize a new database")
    print("  status            Check database status")
    print("  start             Start the database server")
    print("  stop              Stop the database server")
    print("  index             Build ITQ LSH index for nearest neighbor search")
    sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()

    command = sys.argv[1].lower()

    if command in ("init", "initialize"):
        init()
    elif command == "status":
        status()
    elif command == "start":
        start()
    elif command == "stop":
        stop()
    elif command in ("index", "build_index"):
        build_index()
    else:
        print_usage()
