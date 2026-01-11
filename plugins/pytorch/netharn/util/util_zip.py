# -*- coding: utf-8 -*-
"""
Utilities for opening files within a zip archive without explicitly unpacking
it to disk.

TODO:
    - [ ] Move to ubelt?
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import io
import re
import tempfile
import zipfile
import ubelt as ub
from os.path import exists, join


def split_archive(fpath):
    """
    If fpath specifies a file inside a zipfile, it breaks it into two parts the
    path to the zipfile and the internal path in the zipfile.

    fpath = '/'
    split_archive('/a/b/foo.zip/bar.txt')
    split_archive('/a/b/foo.zip/baz/bar.txt')
    split_archive('/a/b/foo.zip/baz/biz.zip/bar.txt')
    split_archive('/a/b/foo.zip/baz/biz.zip/bar.py')
    """
    pat = '(.zip[' + re.escape(os.path.sep) + '/:])'
    parts = re.split(pat, fpath, flags=re.IGNORECASE)
    if len(parts) > 2:
        archivepath = ''.join(parts[:-1])[:-1]
        internal = parts[-1]
    else:
        archivepath = None
        internal = None
    return archivepath, internal


class zopen(ub.NiceRepr):
    """
    Can open a file normally or open a file within a zip file (readonly). Tries
    to read from memory only, but will extract to a tempfile if necessary.

    Example:
        >>> import torch
        >>> dpath = ub.ensure_app_cache_dir('netharn')
        >>> data = torch.FloatTensor([0])
        >>> torch.save(data, join(dpath, 'data.pt'))
        >>> torch.load(join(dpath, 'data.pt'))
        >>> torch.load(open(join(dpath, 'data.pt'), 'rb'))
        >>> torch.load(zopen(join(dpath, 'data.pt'), 'rb'))

    Example:
        >>> import torch
        >>> # Test we can load torch data from a zipfile
        >>> dpath = ub.ensure_app_cache_dir('netharn')
        >>> data = torch.FloatTensor([0])
        >>> datapath = join(dpath, 'data.pt')
        >>> torch.save(data, datapath)
        >>> zippath = join(dpath, 'datazip.zip')
        >>> internal = 'folder/data.pt'
        >>> with zipfile.ZipFile(zippath, 'w') as myzip:
        >>>     myzip.write(datapath, internal)
        >>> fpath = zippath + '/' + internal
        >>> file = zopen(fpath, 'rb', seekable=True)
        >>> data2 = torch.load(file._handle)
        >>> file = zopen(datapath, 'rb', seekable=True)
        >>> data3 = torch.load(file._handle)

    Example:
        >>> # Test we can load json data from a zipfile
        >>> dpath = ub.ensure_app_cache_dir('netharn')
        >>> infopath = join(dpath, 'info.json')
        >>> open(infopath, 'w').write('{"x": "1"}')
        >>> zippath = join(dpath, 'infozip.zip')
        >>> internal = 'folder/info.json'
        >>> with zipfile.ZipFile(zippath, 'w') as myzip:
        >>>     myzip.write(infopath, internal)
        >>> fpath = zippath + '/' + internal
        >>> file = zopen(fpath, 'r')
        >>> import json
        >>> info2 = json.load(file)
        >>> assert info2['x'] == '1'
    """
    def __init__(self, fpath, mode='r', seekable=False):
        self.fpath = fpath
        self.name = fpath
        self.mode = mode
        self._seekable = seekable
        assert 'r' in self.mode
        self._handle = None
        self._zfpath = None
        self._temp_dpath = None
        self._temp_fpath = None
        self._open()

    def __nice__(self):
        if self._zfpath is None:
            return str(self._handle) + ' mode=' + self.mode
        else:
            return '{} in zipfile {}, mode={}'.format(self._handle, self._zfpath, self.mode)

    def __getattr__(self, key):
        # Expose attributes of wrapped handle
        if hasattr(self._handle, key):
            return getattr(self._handle, key)
        raise AttributeError(key)

    def __dir__(self):
        # Expose attributes of wrapped handle
        keyset = set(dir(super(zopen, self)))
        keyset.update(set(self.__dict__.keys()))
        if self._handle is not None:
            keyset.update(set(dir(self._handle)))
        return sorted(keyset)

    def _cleanup(self):
        # print('self._cleanup = {!r}'.format(self._cleanup))
        if not getattr(self, 'closed', True):
            getattr(self, 'close', lambda: None)()
        if self._temp_dpath and exists(self._temp_dpath):
            ub.delete(self._temp_dpath)

    def __del__(self):
        self._cleanup()

    def _open(self):
        _handle = None
        if exists(self.fpath):
            _handle = open(self.fpath, self.mode)
        elif '.zip/' in self.fpath or '.zip' + os.path.sep in self.fpath:
            fpath = self.fpath
            archivefile, internal = split_archive(fpath)
            myzip = zipfile.ZipFile(archivefile, 'r')
            if self._seekable:
                # If we need data to be seekable, then we must extract it to a
                # temporary file first.
                self._temp_dpath = tempfile.mkdtemp()
                temp_fpath = join(self._temp_dpath, internal)
                myzip.extract(internal, self._temp_dpath)
                _handle = open(temp_fpath, self.mode)
            else:
                # Try to load data directly from the zipfile
                _handle = myzip.open(internal, 'r')
                if self.mode == 'rb':
                    data = _handle.read()
                    _handle = io.BytesIO(data)
                elif self.mode == 'r':
                    # FIXME: doesnt always work. handle seems to be closed too
                    # soon in the case util.zopen(module.__file__).read()
                    _handle = io.TextIOWrapper(_handle)
                else:
                    raise KeyError(self.mode)
                self._zfpath = archivefile
        if _handle is None:
            raise IOError('file {!r} does not exist'.format(self.fpath))
        self._handle = _handle

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m netharn.util.util_zip all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
