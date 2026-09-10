"""
Console scripts for the tools provided by KWIVER.

These scripts are used in the wheel setup the environment to kwiver tools and
launch them in a subprocess.
"""

import os
import subprocess
import kwiver
import sys

from pkg_resources import iter_entry_points
from typing import Dict, List

from kwiver.vital import vital_logging
from kwiver.vital.util.initial_plugin_path import get_initial_plugin_path

KWIVER_BIN_DIR = os.path.join(os.path.dirname(os.path.abspath(kwiver.__file__)), "bin")
KWIVER_SUPPORTED_TOOLS = ["kwiver", "plugin_explorer"]
logger = vital_logging.getLogger(__name__)


def _setup_environment() -> Dict:
    """
    Create a dictionary with environment variables for running kwiver tools.

    The dictionary includes appending LD_LIBRARY_PATH, adding path to vital
    logging factory to VITAL_LOGGER_FACTORY, and path to default plugins in
    KWIVER_PLUGIN_PATH.

    Returns:
        Dictionary with environment variables used for running tools
    """
    # Add additional ld libraries
    ld_library_paths = []
    for entry_point in iter_entry_points("kwiver.env.ld_library_path"):
        ld_library_path = entry_point.load()()
        if not os.path.exists(ld_library_path):
            logger.warn(
                f"Invalid path {ld_library_path} specified in {entry_point.name}"
            )
        else:
            ld_library_paths.append(ld_library_path)
    ld_library_path_str = ":".join(ld_library_paths)

    # Add logger factories
    vital_logger_factory = None
    for entry_point in iter_entry_points(
        "kwiver.env.logger_factory", name="vital_log4cplus_logger_factory"
    ):
        logger_factory = entry_point.load()()
        vital_logger_factory = logger_factory

    # Check if LD_LIBRARY_PATH is set to something and append it to the current ld library path
    if os.environ.get("LD_LIBRARY_PATH"):
        ld_library_path_str += os.environ.get("LD_LIBRARY_PATH")

    tool_environment = {
        "LD_LIBRARY_PATH": ld_library_path_str,
        "VITAL_LOGGER_FACTORY": vital_logger_factory,
        "KWIVER_PLUGIN_PATH": get_initial_plugin_path(),
    }
    # Add the remaining environment variables without fiddling with what we have already set
    for env_var_name, env_var_val in os.environ.items():
        if env_var_name not in tool_environment.keys():
            tool_environment[env_var_name] = env_var_val

    return tool_environment


def _kwiver_tools(tool_name: str, args: List[str]) -> int:
    """
    Configure logging, setup environment and run a subprocess with kwiver tool in it.

    Args:
        tool_name: Name of the tool that would be run as a subprocess
        args: Command line argument provided by the user for the tool

    Return:
        Return code for the subprocess that runs the tool
    """
    vital_logging._configure_logging()
    assert (
        tool_name in KWIVER_SUPPORTED_TOOLS
    ), f"Unsupported tool {tool_name} specified"
    tool_environment = _setup_environment()
    tool_path = os.path.join(KWIVER_BIN_DIR, tool_name)
    assert os.path.exists(tool_path), f"Tool {tool_name} not available in {tool_path}"
    args.insert(0, tool_path)
    subprocess_complete = subprocess.run(
        args, shell=False, check=False, env=tool_environment
    )
    return subprocess_complete.returncode


def plugin_explorer() -> None:
    """
    Console script function for plugin_explorer.
    """
    cmd_args = ["--skip-relative"]
    cmd_args.extend(sys.argv[1:])
    raise SystemExit(_kwiver_tools("plugin_explorer", cmd_args))


def kwiver() -> None:
    """
    Console script function for kwiver runner.
    """
    raise SystemExit(_kwiver_tools("kwiver", sys.argv[1:]))
