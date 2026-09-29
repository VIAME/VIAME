import sysconfig
import sys
import os
import itertools

try:
    import distutils.sysconfig as du_sysconfig
except ImportError:
    # distutils was removed in python 3.12. It is only consulted here for
    # LIBDIR, which sysconfig reports as well.
    du_sysconfig = sysconfig


def _library_dirs():
    """
    Where the python library may be, most likely first.

    Debian and Ubuntu report LIBDIR with the multiarch directory already in it
    (`/usr/lib/x86_64-linux-gnu`), and also report `multiarchsubdir`. Joining
    the two, as this used to, looked in `/usr/lib/x86_64-linux-gnu/x86_64-linux-gnu`
    and found nothing -- so LIBDIR is tried as reported first, then with the
    subdirectory, and the static-library config directory last.
    """
    libdir = du_sysconfig.get_config_var("LIBDIR")
    dirs = []
    if libdir:
        dirs.append(libdir)
        if sysconfig.get_config_var("MULTIARCH"):
            masd = sysconfig.get_config_var("multiarchsubdir")
            if masd:
                dirs.append(os.path.join(libdir, masd.lstrip(os.sep)))
    else:
        dirs.append(
            os.path.abspath(
                os.path.join(sysconfig.get_config_var("LIBDEST"), "..", "libs")
            )
        )
    libpl = sysconfig.get_config_var("LIBPL")
    if libpl:
        dirs.append(libpl)
    seen = []
    for d in dirs:
        if d not in seen:
            seen.append(d)
    return seen


def _windows_library():
    """
    The DLL a Windows python loads, or None.

    Windows reports none of LIBDIR, LIBPL, INSTSONAME, LDLIBRARY or
    Py_ENABLE_SHARED, so every branch below falls through to the generic
    search -- which looks for `python3.12.*` under `<prefix>/libs`. The
    library is `python312.dll`: major and minor run together, no dot, no
    `lib` prefix, and it sits beside the interpreter rather than in a
    library directory. `<prefix>/libs` holds `python312.lib`, the import
    library, which is not loadable. So nothing matched and the caller
    logged "Cannot load python library from interpretor or env" on every
    run, on a platform where the symbols were already there through the
    direct link -- an error that meant nothing and hid ones that did.

    The import library is deliberately not a fallback: the caller passes
    this to LoadLibrary.
    """
    if os.name != "nt":
        return None

    name = "python{}{}.dll".format(sys.version_info.major,
                                   sys.version_info.minor)

    for root in (sysconfig.get_config_var("BINDIR"),
                 sys.base_prefix, sys.prefix):
        if not root:
            continue
        candidate = os.path.join(root, name)
        if os.path.exists(candidate):
            return candidate

    return None


def find_python_library():
    """
    Get python library based on sysconfig
    Based on https://github.com/scikit-build/scikit-build/blob/master/skbuild/cmaker.py#L335
    :returns a location python library, or an empty string
    """
    windows = _windows_library()
    if windows:
        return windows

    dirs = _library_dirs()

    # A shared python names its library directly.
    if sysconfig.get_config_var("Py_ENABLE_SHARED"):
        for name in (sysconfig.get_config_var("INSTSONAME"),
                     sysconfig.get_config_var("LDLIBRARY")):
            if not name:
                continue
            for d in dirs:
                candidate = os.path.join(d, name)
                if os.path.exists(candidate):
                    return candidate

    python_library = sysconfig.get_config_var("LIBRARY")
    if not python_library or os.path.splitext(python_library)[1][-2:] == ".a":
        candidate_lib_prefixes = ["", "lib"]
        candidate_implementations = ["python"]
        if hasattr(sys, "pypy_version_info"):
            candidate_implementations = ["pypy-c", "pypy3-c"]
        candidate_extensions = [".lib", ".so", ".a"]
        if sysconfig.get_config_var("WITH_DYLD"):
            candidate_extensions.insert(0, ".dylib")
        candidate_versions = []
        candidate_versions.append("")
        candidate_versions.insert(
            0, str(sys.version_info.major) + "." + str(sys.version_info.minor)
        )
        abiflags = getattr(sys, "abiflags", "")
        candidate_abiflags = [abiflags]
        if abiflags:
            candidate_abiflags.append("")
        for libdir, pre, impl, ext, ver, abi in itertools.product(
            dirs,
            candidate_lib_prefixes,
            candidate_implementations,
            candidate_extensions,
            candidate_versions,
            candidate_abiflags,
        ):
            candidate = os.path.join(libdir, "".join((pre, impl, ver, abi, ext)))
            if os.path.exists(candidate):
                return candidate
        # No valid candidate
        python_library = ""
    return python_library
