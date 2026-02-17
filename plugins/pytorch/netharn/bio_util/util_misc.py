import ubelt as ub
import fnmatch
import re
import os
from os.path import join


def find_files(dpath, glob_pat='*', recursive=True, ignorecase=True,
               followlinks=False):
    """
    Search for files in a directory hierarchy

    Args:
        dpath (str): base directory to search

        glob_pat (str | List[str]): file name pattern or list of patterns in
            glob format

        recursive (bool, default=True): recursive flag

        followlinks (bool, default=False): follows symlinks

    Yields:
        str: matching file paths

    References:
        https://stackoverflow.com/questions/8151300/ignore-case-in-glob-on-linux

    Example:
        >>> import kwimage
        >>> from os.path import dirname
        >>> ignorecase = True
        >>> dpath = dirname(kwimage.__file__)
        >>> glob_pat = '*.So'
        >>> fpaths = list(find_files(dpath, glob_pat))
        >>> print('fpaths = {}'.format(ub.urepr(fpaths, nl=1)))

        >>> glob_pat = ['*.So', '*py']
        >>> fpaths = list(find_files(dpath, glob_pat))
        >>> print('fpaths = {}'.format(ub.urepr(fpaths, nl=1)))
    """
    flags = 0
    if ignorecase:
        flags |= re.IGNORECASE

    if ub.iterable(glob_pat):
        regex_pat = '|'.join([fnmatch.translate(p) for p in glob_pat])
    else:
        regex_pat = fnmatch.translate(glob_pat)
    regex = re.compile(regex_pat, flags=flags)
    # note: os.walk is faster than os.listdir
    for root, dirs, files in os.walk(dpath, followlinks=followlinks):
        for fname in files:
            if regex.match(fname):
                fpath = join(root, fname)
                yield fpath
        if not recursive:
            break


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/bioharn/bioharn/util/util_misc.py all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
