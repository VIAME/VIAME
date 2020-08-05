import sysconfig
import distutils.sysconfig as du_sysconfig
import sys
import os
import itertools

def find_python_library():
    """
    Get python library based on sysconfig
    Based on https://github.com/scikit-build/scikit-build/blob/master/skbuild/cmaker.py#L335
    :returns a location python library
    """
    python_library = sysconfig.get_config_var('LIBRARY')
    if (not python_library or os.path.splitext(python_library)[1][-2:] == '.a'):
        candidate_lib_prefixes = ['', 'lib']
        candidate_implementations = ['python']
        if hasattr(sys, "pypy_version_info"):
            candidate_implementations = ['pypy-c', 'pypy3-c']
        candidate_extensions = ['.lib', '.so', '.a']
        if sysconfig.get_config_var('WITH_DYLD'):
            candidate_extensions.insert(0, '.dylib')
        candidate_versions = []
        candidate_versions.append('')
        candidate_versions.insert(0, str(sys.version_info.major) +
                                     "." + str(sys.version_info.minor))
        abiflags = getattr(sys, 'abiflags', '')
        candidate_abiflags = [abiflags]
        if abiflags:
            candidate_abiflags.append('')
        # Ensure the value injected by virtualenv is
        # returned on windows.
        # Because calling `sysconfig.get_config_var('multiarchsubdir')`
        # returns an empty string on Linux, `du_sysconfig` is only used to
        # get the value of `LIBDIR`.
        libdir = du_sysconfig.get_config_var('LIBDIR')
        if sysconfig.get_config_var('MULTIARCH'):
            masd = sysconfig.get_config_var('multiarchsubdir')
            if masd:
                if masd.startswith(os.sep):
                    masd = masd[len(os.sep):]
                    libdir = os.path.join(libdir, masd)
        if libdir is None:
            libdir = os.path.abspath(os.path.join(
                sysconfig.get_config_var('LIBDEST'), "..", "libs"))
        no_valid_candidate = True
        for (pre, impl, ext, ver, abi) in itertools.product(candidate_lib_prefixes,
                                                            candidate_implementations,
                                                            candidate_extensions,
                                                            candidate_versions,
                                                            candidate_abiflags):
            candidate = os.path.join(libdir, ''.join((pre, impl, ver, abi, ext)))
            if os.path.exists(candidate):
                python_library = candidate
                no_valid_candidate = False
                break
        # If there is not valid candidate then set the python_library is empty
        if no_valid_candidate:
            python_library = ""
    return python_library
