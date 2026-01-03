"""
SMQTK Data Element - Abstract data container interface.
"""
import abc
import hashlib
import io
import mimetypes
import os
import os.path as osp
import re
import tempfile

import six

from ..exceptions import InvalidUriError, NoUriResolutionError, ReadOnlyError
from ..utils import SmqtkObject, safe_create_dir, safe_file_write
from ..utils.plugin import Pluggable


MIMETYPES = mimetypes.MimeTypes()


@six.add_metaclass(abc.ABCMeta)
class DataElement(SmqtkObject, Pluggable):
    """
    Abstract interface for a byte data container.
    """

    @classmethod
    def from_uri(cls, uri):
        """
        Construct a new instance based on the given URI.
        """
        raise NoUriResolutionError()

    def __init__(self):
        super(DataElement, self).__init__()
        self._temp_filepath_stack = []

    __hash__ = None

    def __del__(self):
        self.clean_temp()

    def __eq__(self, other):
        return isinstance(other, DataElement) and \
               self.get_bytes() == other.get_bytes()

    def __ne__(self, other):
        return not (self == other)

    @abc.abstractmethod
    def __repr__(self):
        return self.__class__.__name__

    def _write_new_temp(self, d):
        """Write bytes to a new temp file."""
        if d:
            safe_create_dir(d)
        ext = MIMETYPES.guess_extension(self.content_type() or '')
        if ext in {'.jpe', '.jfif'}:
            ext = '.jpg'
        fd, fp = tempfile.mkstemp(suffix=ext or '', dir=d)
        os.close(fd)
        with open(fp, 'wb') as f:
            f.write(self.get_bytes())
        return fp

    def md5(self):
        """Get the MD5 checksum of this element's binary content."""
        return hashlib.md5(self.get_bytes()).hexdigest()

    def sha1(self):
        """Get the SHA1 checksum of this element's binary content."""
        return hashlib.sha1(self.get_bytes()).hexdigest()

    def write_temp(self, temp_dir=None):
        """Write this data's bytes to a temporary file on disk."""
        # Clear out paths that don't exist
        self._temp_filepath_stack = [
            fp for fp in self._temp_filepath_stack if osp.isfile(fp)
        ]

        if temp_dir:
            abs_temp_dir = osp.abspath(osp.expanduser(temp_dir))
            for tf in self._temp_filepath_stack:
                if osp.dirname(tf) == abs_temp_dir:
                    return tf
            self._temp_filepath_stack.append(self._write_new_temp(temp_dir))
        elif not self._temp_filepath_stack:
            self._temp_filepath_stack.append(self._write_new_temp(None))

        return self._temp_filepath_stack[-1]

    def clean_temp(self):
        """Clean any temporary files created by this element."""
        if len(self._temp_filepath_stack):
            for fp in self._temp_filepath_stack:
                if os.path.isfile(fp):
                    os.remove(fp)
            self._temp_filepath_stack = []

    def uuid(self):
        """UUID for this data element."""
        return self.sha1()

    def to_buffered_reader(self):
        """Wrap this element's bytes in a BufferedReader instance."""
        return io.BufferedReader(io.BytesIO(self.get_bytes()))

    def is_read_only(self):
        """Return if this element can only be read from."""
        return not self.writable()

    @abc.abstractmethod
    def content_type(self):
        """Return standard type/subtype string for this data element."""

    @abc.abstractmethod
    def is_empty(self):
        """Check if this element contains no bytes."""

    @abc.abstractmethod
    def get_bytes(self):
        """Get the bytes for this data element."""

    @abc.abstractmethod
    def writable(self):
        """Return if this instance supports setting bytes."""

    @abc.abstractmethod
    def set_bytes(self, b):
        """Set bytes to this data element."""
        if not self.writable():
            raise ReadOnlyError("This %s element is read only." % self)


STR_NONE_TYPES = six.string_types + (type(None),)


class DataFileElement(DataElement):
    """
    File-based data element.
    """

    FILE_URI_RE = re.compile("^(?:file://)?(/?[^/]+(?:/[^/]+)*)$")

    @classmethod
    def is_usable(cls):
        return True

    @classmethod
    def from_uri(cls, uri):
        """Construct a new instance based on the given URI."""
        path_match = cls.FILE_URI_RE.match(uri)

        if path_match is None:
            raise InvalidUriError(uri, "Malformed URI")

        path = path_match.group(1)

        if uri.startswith("file://") and not osp.isabs(path):
            raise InvalidUriError(uri, "Found file:// prefix, but path was not "
                                       "absolute")

        return DataFileElement(path)

    def __init__(self, filepath, readonly=False, explicit_mimetype=None):
        """Create a new FileElement."""
        super(DataFileElement, self).__init__()

        assert isinstance(filepath, six.string_types), \
            "File path must be a string."
        assert isinstance(explicit_mimetype, STR_NONE_TYPES), \
            "Explicit mimetype must either be a string or None."

        self._filepath = osp.expanduser(filepath)
        self._readonly = bool(readonly)
        self._explicit_mimetype = explicit_mimetype

        self._content_type = explicit_mimetype
        if not self._content_type:
            self._content_type = mimetypes.guess_type(filepath)[0]

    def __repr__(self):
        return super(DataFileElement, self).__repr__() + \
            "{filepath: %s, readonly: %s, explicit_mimetype: %s}" \
            % (self._filepath, self._readonly, self._explicit_mimetype)

    def get_config(self):
        return {
            "filepath": self._filepath,
            "readonly": self._readonly,
            "explicit_mimetype": self._explicit_mimetype,
        }

    def content_type(self):
        """Return standard type/subtype string for this data element."""
        return self._content_type

    def is_empty(self):
        """Check if this element contains no bytes."""
        return not osp.exists(self._filepath) or \
            osp.getsize(self._filepath) == 0

    def get_bytes(self):
        """Get the byte stream for this data element."""
        return (not self.is_empty() and open(self._filepath, 'rb').read()) or b""

    def writable(self):
        """Return if this instance supports setting bytes."""
        return not self._readonly

    def set_bytes(self, b):
        """Set bytes to this data element."""
        if not self._readonly:
            safe_file_write(self._filepath, b)
        else:
            raise ReadOnlyError("This file element is read only.")

    def write_temp(self, temp_dir=None):
        """Write this data's bytes to a temporary file."""
        if temp_dir:
            abs_temp_dir = osp.abspath(osp.expanduser(temp_dir))
            if abs_temp_dir != osp.dirname(self._filepath):
                return super(DataFileElement, self).write_temp(temp_dir)
        return self._filepath

    def clean_temp(self):
        """Clean any temporary files."""
        return super(DataFileElement, self).clean_temp()


DATA_ELEMENT_CLASS = DataFileElement


def get_data_element_impls(reload_modules=False):
    """
    Return available DataElement implementations.
    """
    return {
        'DataFileElement': DataFileElement,
    }


__all__ = [
    'DataElement',
    'DataFileElement',
    'get_data_element_impls',
]
