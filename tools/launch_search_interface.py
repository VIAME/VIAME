#!/usr/bin/env python

# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Launch the VIAME search/query GUI interface."""

import argparse
import atexit
import glob
import os
import shutil
import string
import sys
import tempfile
import urllib.parse as urlparse

import database_tool

DIV = '\\' if os.name == 'nt' else '/'
DEBUG_MODE = False

temp_dir = tempfile.mkdtemp(prefix='viqui-tmp')
atexit.register(lambda: shutil.rmtree(temp_dir))


def _format_path(path):
    if os.name == 'nt':
        return path.replace('\\', '/').replace(' ', '%20')
    return path


def _list_files(folder, extension):
    return glob.glob(os.path.join(folder, f'*{extension}'))


def _get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def _find_file(filename):
    if os.path.exists(filename):
        return os.path.abspath(filename)
    alt_path = os.path.join(_get_script_path(), filename)
    if os.path.exists(alt_path):
        return alt_path
    print(f"Unable to find {filename}")
    sys.exit(1)


def _execute_command(cmd):
    if os.name == 'nt':
        return os.system(cmd)
    return os.system(f'/bin/bash -c "{cmd}"')


def _make_uri(scheme='file', authority='', path='', query=''):
    return urlparse.urlunsplit([scheme, authority, _format_path(path), query, ''])


def _is_valid_database(options):
    if not os.path.exists(options.input_dir):
        print(f"\nERROR: \"{options.input_dir}\" directory does not exist, "
              "was your create_index call successful?\n")
        return False
    if len(glob.glob(os.path.join(options.input_dir, "*.index"))) == 0:
        print(f"\nERROR: \"{options.input_dir}\" is empty, "
              "was your create_index call successful?\n")
        return False
    return True


def _get_gui_cmd():
    if os.name == 'nt':
        return 'viqui.exe '
    if DEBUG_MODE:
        return 'gdb --args viqui '
    return 'viqui '


def _get_import_config_args(config_files):
    return ' '.join(f'--import-config {f}' for f in config_files) + ' '


def _get_query_server_uri(options):
    query_pipeline = os.path.abspath(_find_file(options.query_pipe))
    query = f'Pipeline={_format_path(query_pipeline)}'
    return _make_uri(scheme='kip', query=query)


def _create_archive_file(options):
    fd, name = tempfile.mkstemp(prefix='viqui-archive-', suffix='.archive',
                                text=True, dir=temp_dir)
    with os.fdopen(fd, 'w') as f:
        f.write('archive1\n')
        for index_file in _list_files(options.input_dir, '.index'):
            f.write(os.path.abspath(index_file) + "\n")
    return name


def _get_predefined_query_dir(options):
    query_dir = options.predefined_dir
    if os.path.exists(query_dir) and not os.path.isdir(query_dir):
        print(f'{query_dir} is not a directory.')
        sys.exit(1)
    return query_dir


CONFIG_TEMPLATE = string.Template('''
[General]
QueryVideoUri = $query_video_uri
VideoProviderUris=$video_provider_uri
QueryCacheUri=$query_cache_uri
QueryServerUri=$query_server_uri
PredefinedQueryUri=$predefined_query_uri
''')

CONFIG_TEMPLATE_NO_CACHE = string.Template('''
[General]
QueryVideoUri = $query_video_uri
VideoProviderUris=$video_provider_uri
QueryServerUri=$query_server_uri
PredefinedQueryUri=$predefined_query_uri
''')


def _create_constructed_config(options):
    query_server_uri = _get_query_server_uri(options)
    query_video_uri = _make_uri(path=os.path.abspath(options.query_dir))
    archive_file = _create_archive_file(options)
    video_provider_uri = _make_uri(path=archive_file)
    predefined_query_dir = _get_predefined_query_dir(options)
    predefined_query_uri = _make_uri(path=os.path.abspath(predefined_query_dir))

    fd, name = tempfile.mkstemp(prefix='viqui-config-', suffix='.conf',
                                text=True, dir=temp_dir)
    with os.fdopen(fd, 'w') as f:
        if options.cache_dir != "disabled":
            query_cache_uri = _make_uri(path=os.path.abspath(options.cache_dir))
            f.write(CONFIG_TEMPLATE.substitute(
                query_video_uri=query_video_uri,
                video_provider_uri=video_provider_uri,
                query_cache_uri=query_cache_uri,
                query_server_uri=query_server_uri,
                predefined_query_uri=predefined_query_uri))
        else:
            f.write(CONFIG_TEMPLATE_NO_CACHE.substitute(
                query_video_uri=query_video_uri,
                video_provider_uri=video_provider_uri,
                query_server_uri=query_server_uri,
                predefined_query_uri=predefined_query_uri))
    return name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch Query GUI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-d", dest="input_dir", default="database",
                        help="Input directory containing videos and results")
    parser.add_argument("-c", dest="context_file",
                        default=f"gui-params{DIV}context_view_bluemarble_low_res.kst",
                        help="GUI context file for display on left panel")
    parser.add_argument("-s", dest="style", default="",
                        help="Optional GUI style option, blank for default")
    parser.add_argument("-e", "--engineer", action="store_true",
                        dest="engineer_mode", default=False,
                        help="Turn on the engineer UI (add developer options)")
    parser.add_argument("-qp", dest="query_pipe",
                        default=f"pipelines{DIV}query_retrieval_and_iqr.pipe",
                        help="Pipeline for performing new system queries")
    parser.add_argument("-qd", dest="query_dir",
                        default=f"database{DIV}Queries",
                        help="Directory for writing new queries and configs to")
    parser.add_argument("-cd", dest="cache_dir", default="disabled",
                        help="Directory for caching repeated queries")
    parser.add_argument("-pd", dest="predefined_dir",
                        default=f"pipelines{DIV}predefined_queries",
                        help="Predefined query directory, if present")
    parser.add_argument("-theme", dest="gui_theme",
                        default=f"gui-params{DIV}view_color_settings.ini",
                        help="GUI theme settings file")
    parser.add_argument("--no-reconfig", dest="no_reconfig", action="store_true",
                        help="Do not run any reconfiguration of the GUI")
    parser.add_argument("--debug", dest="debug", action="store_true",
                        help="Run with debugger attached to process")

    args = parser.parse_args()

    if not _is_valid_database(args):
        sys.exit(1)

    # Create required directories
    if not os.path.exists(args.query_dir):
        os.makedirs(args.query_dir)
    if args.cache_dir != "disabled" and not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    if args.debug:
        DEBUG_MODE = True

    # Build command line
    command = _get_gui_cmd() + \
              f'--add-layer "{_get_script_path()}{DIV}{args.context_file}" '

    if not args.no_reconfig:
        if args.gui_theme:
            command += f'--theme "{_find_file(args.gui_theme)}" '
        command += "--ui engineering " if args.engineer_mode else "--ui analyst "
        command += _get_import_config_args([_create_constructed_config(args)])

    # Make sure database is online
    sql_dir = database_tool.SQL_DIR
    if not os.path.exists(sql_dir):
        print(f"\nERROR: Database directory \"{sql_dir}\" does not exist.")
        print("Please run create_index first to initialize the database.")
        sys.exit(1)

    # Check if server is already running for this database
    if not database_tool.status(quiet=True):
        # Not running - stop any other PostgreSQL instances that may be holding the port
        print("Stopping any existing database instances...")
        database_tool.stop(quiet=True)

        # Wait for port to become available
        if not database_tool._wait_for_port_available(timeout=10):
            print("Warning: Port 5432 may still be in use, attempting to start anyway...")

        # Clean up any stale lock files from previous sessions
        postmaster_pid = os.path.join(sql_dir, "postmaster.pid")
        if os.path.exists(postmaster_pid):
            print("Cleaning up stale database lock files...")
            os.remove(postmaster_pid)

        # Now try to start the database
        print("Starting database...")
        if not database_tool.start(quiet=False):
            print(f"\nERROR: Database in \"{sql_dir}\" failed to start.")
            print("Check database/SQL_Log_File for details.")
            print("Try running: database_tool.py init")
            sys.exit(1)

    print("\nLaunching search GUI. When finished, make sure this console is closed.\n")
    _execute_command(command)
