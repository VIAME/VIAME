import logging
import os
import sys
from pathlib import Path


PYTHON_PLUGIN_ENTRYPOINT = "viame.python_plugin_registration"
CPP_SEARCH_PATHS_ENTRYPOINT = "viame.cpp_search_paths"


_LOGGING_ENVIRON_VAR = "KWIVER_PYTHON_DEFAULT_LOG_LEVEL"


def _preload_libpython() -> None:
    """Make libpython findable before any extension module loads.

    Every extension here, and libviame itself, carries an explicit
    `NEEDED libpython3.X.so.1.0`. That is unusual -- an extension normally
    leaves python symbols undefined and resolves them from the interpreter
    already in memory -- and it means the loader has to *find* libpython by
    name when the first native submodule is imported.

    On a distribution python it does: Ubuntu keeps libpython3.10 in the
    ldconfig cache. A standalone interpreter -- pyenv, conda, uv,
    python.org -- keeps it in its own lib directory, which is on nobody's
    search path, and every `import viame.types` fails with

        ImportError: libpython3.12.so.1.0: cannot open shared object file

    while a bare `import viame` succeeds, because this file is pure python.

    Loading it here by absolute path registers its soname with the loader,
    so the NEEDED entries of everything imported afterwards resolve against
    this copy. Silent and best-effort: where libpython is already reachable
    this changes nothing, and a statically linked interpreter has no file to
    load and does not need one.
    """
    if os.name != "posix":
        return

    import sysconfig

    names = []
    ldlib = sysconfig.get_config_var("INSTSONAME") or sysconfig.get_config_var("LDLIBRARY")
    if ldlib:
        names.append(ldlib)
    names.append("libpython{}.{}.so.1.0".format(*sys.version_info[:2]))

    for var in ("LIBDIR", "LIBPL"):
        directory = sysconfig.get_config_var(var)
        if not directory:
            continue
        for name in names:
            candidate = Path(directory) / name
            if candidate.is_file():
                try:
                    import ctypes
                    ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                    return
                except OSError:
                    pass


def _add_windows_dll_directories() -> None:
    """Windows' answer to the `$ORIGIN` RUNPATH the ELF builds carry.

    A wheel resolves CUDA out of the `nvidia-*` wheels pip installed beside
    it. On Linux that is a RUNPATH baked into each binary; Windows has no
    RUNPATH at all, and since 3.8 will not search PATH for a dependent DLL of
    an extension module either. `os.add_dll_directory` is what is left.

    Both layouts are offered because they differ by CUDA major -- cu12 keeps
    a directory per component, cu13 consolidated -- and a directory that is
    not there costs nothing. `bin` rather than `lib`: the windows nvidia
    wheels put their DLLs in `nvidia/<component>/bin` and only the import
    libraries in `lib`.

    Silent by design. A CPU-only install has none of these and must not be
    made to look broken by it.
    """
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    roots = []
    for entry in sys.path:
        nvidia = Path(entry) / "nvidia"
        if nvidia.is_dir():
            roots.append(nvidia)

    # `<env>/Library/bin` is where this wheel's own DLLs land, beside the
    # tools, mirroring `{data}/lib` on Linux.
    prefix = Path(sys.prefix)
    for extra in (prefix / "Library" / "bin", prefix / "bin"):
        if extra.is_dir():
            try:
                os.add_dll_directory(str(extra))
            except OSError:
                pass

    for root in roots:
        for component in sorted(root.iterdir()):
            for candidate in (component / "bin", component / "lib"):
                if candidate.is_dir():
                    try:
                        os.add_dll_directory(str(candidate))
                    except OSError:
                        pass


_preload_libpython()
_add_windows_dll_directories()


def _logging_onetime_init() -> None:
    """
    One-time initialize viame-module-scope logging level.

    This will default to the WARNING level unless the environment variable
    "KWIVER_PYTHON_DEFAULT_LOG_LEVEL" is set to a valid case-insensitive value
    (see the map below).

    This does NOT create a logging formatter. This is left to the discretion of
    the application.
    """
    if not hasattr(_logging_onetime_init, "called"):
        # Pull logging from environment variable if set
        llevel = logging.WARN
        if _LOGGING_ENVIRON_VAR in os.environ:
            llevel_str = os.environ[_LOGGING_ENVIRON_VAR].lower()
            # error warn info debug trace
            m = {
                "error": logging.ERROR,
                "warn": logging.WARN,
                "info": logging.INFO,
                "debug": logging.DEBUG,
                "trace": 1,
            }
            if llevel_str in m:
                llevel = m[llevel_str]
            else:
                logging.getLogger("viame").warning(
                    f"KWIVER python logging level value set but did not match "
                    f'a valid value. Was given: "{llevel_str}". '
                    f"Must be one of: {list(m.keys())}. Defaulting to warning "
                    f"level."
                )
        logging.getLogger(__name__).setLevel(llevel)
        # Mark this one-time logic as invoked to mater calls are idempotent.
        _logging_onetime_init.called = True
    else:
        logging.getLogger(__name__).debug(
            "Logging one-time setup already called, doing nothing."
        )


def _add_library_paths() -> None:
    # For Python >=3.8  we need to explicitly add paths where dll will be imported from.
    if sys.version_info >= (3, 8) and sys.platform == "win32":
        paths = [
            # in the build directory __file__ is at Lib\site_packages\viame and viame dlls in toplevel bin
            Path(__file__).parents[3].absolute() / "bin",
            # in a a wheel __file__ is at Lib\site_packages\viame and viame dlls in  the same level bin
            Path(__file__).parents[0].absolute() / "bin",
        ]
        for path in paths:
            if path.exists():
                os.add_dll_directory(str(path))
                logging.getLogger(__name__).debug(f"Adding {path} to dll search paths")
        # Add viame.libs to PATH when using wheels
        # This ensures VIAME's C++ plugin loader (LoadLibraryW) can find DLLs
        # since it doesn't use Python's os.add_dll_directory() search paths
        path = (Path(__file__).parents[1] / "viame.libs").resolve()
        if path.exists():
            os.environ["PATH"] = f"{str(path)}{os.pathsep}{os.environ['PATH']}"
            logging.getLogger(__name__).debug(f"Adding {path} to PATH")


def _setup_projdb_path() -> None:
    """Set path to proj.db file. Call to proj library require access to this
    file. Ideally we should use an API call to the proj arrow to set this path via
    `proj_context_set_search_paths()`
    See also https://proj.org/en/6.3/resource_files.html#where-are-proj-resource-files-looked-for
    """
    if "PROJ_LIB" not in os.environ:
        path = str(Path(__file__).parents[0] / "share")
        os.environ["PROJ_LIB"] = path
        logging.getLogger(__name__).debug(f"Setting PROJ_LIB to {path}")


_logging_onetime_init()
_add_library_paths()
_setup_projdb_path()

# Wait for environment to be properly set up before importing submodules
