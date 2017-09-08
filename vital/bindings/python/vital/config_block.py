"""
ckwg +31
Copyright 2015-2016 by Kitware, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither name of Kitware, Inc. nor the names of any contributors may be used
   to endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

==============================================================================

Interface to VITAL config_block class.

"""
import ctypes

from vital.exceptions.config_block import (
    VitalConfigBlockNoSuchValueException,
    VitalConfigBlockReadOnlyException,
)
from vital.exceptions.config_block_io import (
    VitalConfigBlockIoException,
    VitalConfigBlockIoFileNotFoundException,
    VitalConfigBlockIoFileNotReadException,
    VitalConfigBlockIoFileNotParsed,
    VitalConfigBlockIoFileWriteException,
)
from vital.util import VitalObject, VitalErrorHandle, free_void_ptr

import os
from six.moves import range
import tempfile


class ConfigBlock (VitalObject):
    """
    vital::config_block interface class
    """

    # ConfigBlock Constants -- initialized later in file
    BLOCK_SEP = None
    GLOBAL_VALUE = None

    @classmethod
    def from_file(cls, filepath):
        """
        Return a new ConfigBlock object based on the given configuration
        file

        :raises VitalBaseException: Error occurred in reading from configuration
            file path given.

        :param filepath: Path to a configuration file
        :type filepath: str

        :return: New ConfigBlock instance
        :rtype: ConfigBlock

        """
        cb_read = cls.VITAL_LIB.vital_config_block_file_read
        cb_read.argtypes = [ctypes.c_char_p, VitalErrorHandle.C_TYPE_PTR]
        cb_read.restype = cls.C_TYPE_PTR
        with VitalErrorHandle() as eh:
            eh.set_exception_map({
                -1: VitalConfigBlockIoException,
                1: VitalConfigBlockIoFileNotFoundException,
                2: VitalConfigBlockIoFileNotReadException,
                3: VitalConfigBlockIoFileNotParsed
            })
            cptr = cb_read(filepath, eh)
        return cls(from_cptr=cptr)

    def __init__(self, name=None, from_cptr=None):
        """
        Create a new, empty configuration instance.

        :param name: Optional name for the configuration
        :type name: str

        """
        super(ConfigBlock, self).__init__(from_cptr, name)

    def _new(self, name):
        if name:
            cb_new = self.VITAL_LIB.vital_config_block_new_named
            cb_new.argtypes = [ctypes.c_char_p]
            cb_new.restype = self.C_TYPE_PTR
            return cb_new(str(name))
        else:
            cb_new = self.VITAL_LIB.vital_config_block_new
            cb_new.restype = self.C_TYPE_PTR
            return cb_new()

    def _destroy(self):
        if self.c_pointer:
            self.VITAL_LIB.vital_config_block_destroy(self)

    @property
    def name(self):
        """
        :return: The name assigned to this ConfigBlock instance.
        :rtype: str
        """
        return self._call_cfunc(
            'vital_config_block_get_name',
            [self.C_TYPE_PTR], [self],
            ctypes.c_char_p
        )

    def subblock(self, key):
        """
        Get a new ConfigBlock that is deep copy of this ConfigBlock
        starting at the given key.

        :return: New ConfigBlock
        :rtype: ConfigBlock

        """
        cptr = self._call_cfunc(
            'vital_config_block_subblock',
            [self.C_TYPE_PTR, ctypes.c_char_p],
            [self, key],
            self.C_TYPE_PTR
        )
        return ConfigBlock(from_cptr=cptr)

    def subblock_view(self, key):
        """
        Get a new ConfigBlock that is a view into this ConfigBlock
        starting at the given key.

        Modifications on the returned ConfigBlock are also reflected in
        this instance.

        :return: New ConfigBlock instance with this instance as its parent.
        :rtype: ConfigBlock

        """
        cptr = self._call_cfunc(
            'vital_config_block_subblock_view',
            [self.C_TYPE_PTR, ctypes.c_char_p],
            [self, key],
            self.C_TYPE_PTR
        )
        return ConfigBlock(from_cptr=cptr)

    def get_value(self, key, default=None):
        """ Get the string value for a key.

        The given key may not exist in this configuration. If no default was
        given, then an exception is raised. Otherwise, the provided default
        value is returned.

        :param key: The index of the configuration value to retrieve.
        :type key: str

        :param default: Optional default value that will be returned if the
            given is not found in this configuration.
        :type default: str

        :return: The string value stored within the configuration
        :rtype: str

        :raises VitalConfigBlockNoSuchValueException: the given key doesn't
            exist in the configuration and no default was provided.

        """
        c_arg_t = [self.C_TYPE_PTR, ctypes.c_char_p]
        c_arg_v = [self, key]
        c_arg_r = ctypes.c_void_p
        ex_map = {1: VitalConfigBlockNoSuchValueException}

        if default is None:
            v = self._call_cfunc(
                'vital_config_block_get_value',
                c_arg_t, c_arg_v, c_arg_r, ex_map
            )
        else:
            v = self._call_cfunc(
                'vital_config_block_get_value_default',
                c_arg_t + [ctypes.c_char_p],
                c_arg_v + [default],
                c_arg_r,
                ex_map
            )

        s = ctypes.c_char_p(v).value
        free_void_ptr(v)
        return s

    def get_value_bool(self, key, default=None):
        """ Get the boolean value for a key

        The given key may not exist in this configuration. If no default was
        given, then an exception is raised. Otherwise, the provided default
        value is returned.

        :param key: The index of the configuration value to retrieve.
        :type key: str

        :param default: Optional default value that will be returned if the
            given is not found in this configuration.
        :type default: bool

        :return: The boolean value stored within the configuration
        :rtype: bool

        :raises VitalConfigBlockNoSuchValueException: the given key doesn't
            exist in the configuration and no default was provided.

        """
        c_arg_t = [self.C_TYPE_PTR, ctypes.c_char_p]
        c_arg_v = [self, key]
        c_arg_r = ctypes.c_bool
        ex_map = {1: VitalConfigBlockNoSuchValueException}

        if default is None:
            return self._call_cfunc(
                'vital_config_block_get_value_bool',
                c_arg_t, c_arg_v, c_arg_r, ex_map
            )
        else:
            return self._call_cfunc(
                'vital_config_block_get_value_default_bool',
                c_arg_t + [ctypes.c_bool],
                c_arg_v + [default],
                c_arg_r, ex_map
            )

    def get_description(self, key):
        """
        Get the string description for a given key.

        :raises VitalConfigBlockNoSuchValueException: the given key doesn't
            exist in the configuration.

        :param key: The name of the parameter to get the description of.
        :type key: str

        :returns: The string description of the give key.
        :rtype: str or None

        """
        v = self._call_cfunc(
            'vital_config_block_get_description',
            [self.C_TYPE_PTR, ctypes.c_char_p],
            [self, key],
            ctypes.c_void_p,
            {1: VitalConfigBlockNoSuchValueException}
        )
        s = ctypes.c_char_p(v).value
        free_void_ptr(v)
        return s

    def set_value(self, key, value, description=None):
        """
        Set a string value within the configuration.

        If the provided key has been marked as read-only, nothing is set.

        If this key already exists, has a description and no new description
        was passed with this call, the previous description is retained. We
        assume that the previous description is still valid and this a value
        overwrite. If it is intended for the description to also be overwritten,
        an unset_value call should be performed on the key first, and then this
        set_value call.

        :param key: The index of the configuration value to set.
        :type key: str

        :param value: The value to set for the \p key.

        :param description: Optional description to associate with the key.
        :type description: str

        """
        if description is None:
            self._call_cfunc(
                'vital_config_block_set_value',
                [self.C_TYPE_PTR, ctypes.c_char_p, ctypes.c_char_p],
                [self, str(key), str(value)],
                exception_map={
                    1: VitalConfigBlockReadOnlyException
                }
            )
        else:
            self._call_cfunc(
                'vital_config_block_set_value_descr',
                [self.C_TYPE_PTR, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p],
                [self, str(key), str(value), str(description)],
                exception_map={
                    1: VitalConfigBlockReadOnlyException
                }
            )

    def unset_value(self, key):
        """
        Remove a value from the configuration

        If the provided key has been marked as read-only, nothing is unset.

        :param key: The index of the configuration value to unset
        :type key: str

        """
        self._call_cfunc(
            'vital_config_block_unset_value',
            [self.C_TYPE_PTR, ctypes.c_char_p],
            [self, str(key)],
            exception_map={
                1: VitalConfigBlockReadOnlyException,
                2: VitalConfigBlockNoSuchValueException,
            }
        )

    def is_read_only(self, key):
        """
        Query if a value is read-only.

        :param key: Key to check
        :type key: str

        :return: True if the given key has been marked as read-only, and false
            otherwise
        :rtype: bool

        """
        return self._call_cfunc(
            'vital_config_block_is_read_only',
            [self.C_TYPE_PTR, ctypes.c_char_p],
            [self, key],
            ctypes.c_bool
        )

    def mark_read_only(self, key):
        """
        Mark the given key as read-only

        This provided key is marked as read-only even if it doesn't currently
        exist in the given config_block instance.

        :param key: The key to mark as read-only.
        :type key: str

        """
        self._call_cfunc(
            'vital_config_block_mark_read_only',
            [self.C_TYPE_PTR, ctypes.c_char_p],
            [self, key]
        )

    def merge_config(self, other):
        """
        Merge the values in given ConfigBlock into the this instance.

        NOTE: Any values currently set within \c *this will be overwritten if
        conflicts occur.

        :param other: Other ConfigBlock instance whose key/value pairs are
            to be merged into this instance.
        :type other: ConfigBlock

        """
        self._call_cfunc(
            'vital_config_block_merge_config',
            [self.C_TYPE_PTR, self.C_TYPE_PTR],
            [self, other]
        )

    def has_value(self, key):
        """
        Check if a value exists for key in this configuration

        :param key: The index of the configuration value to check.
        :type key: str

        :returns: Whether the key exists or not
        :rtype: bool

        """
        return self._call_cfunc(
            'vital_config_block_has_value',
            [self.C_TYPE_PTR, ctypes.c_char_p],
            [self, key],
            ctypes.c_bool
        )

    def available_keys(self):
        """
        Return a list of available keys in this configuration instance.
        """
        cb_ak = self.VITAL_LIB.vital_config_block_available_values
        sl_free = self.VITAL_LIB.vital_common_free_string_list

        cb_ak.argtypes = [self.C_TYPE_PTR, ctypes.POINTER(ctypes.c_uint),
                          ctypes.POINTER(ctypes.POINTER(ctypes.c_char_p))]
        sl_free.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_char_p)]

        length = ctypes.c_uint(0)
        keys = ctypes.POINTER(ctypes.c_char_p)()
        cb_ak(self, ctypes.byref(length), ctypes.byref(keys))

        # Constructing return array
        r = []
        for i in range(length.value):
            r.append(keys[i])

        # Free allocated key listing
        sl_free(length, keys)

        return r

    def read(self, filepath):
        """
        Update this configuration instance with the configuration at the given
        filepath.

        :param filepath: Path to a configuration file
        :type filepath: str

        """
        cb = ConfigBlock.from_file(filepath)
        self.merge_config(cb)

    def write(self, filepath):
        """
        Output this configuration to the specified file path

        :raises VitalBaseException: Error occurred in writing configuration to
            the given file path.

        :param filepath: Output file path.

        """
        cb_write = self.VITAL_LIB.vital_config_block_file_write
        cb_write.argtypes = [self.C_TYPE_PTR, ctypes.c_char_p,
                             VitalErrorHandle.C_TYPE_PTR]
        with VitalErrorHandle() as eh:
            eh.set_exception_map({
                -1: VitalConfigBlockIoException,
                1: VitalConfigBlockIoFileWriteException
            })

            cb_write(self, filepath, eh)

    def as_dict(self):
        """
        Return this configuration as a dictionary mapping available keys to
        their string values.

        :return: Dictionary of keys/value pairs
        :rtype: dict of (str, str)

        """
        d = {}
        for k in self.available_keys():
            d[k] = self.get_value(k)
        return d

    def as_string(self):
        """
        Return stringification of this configuration, which is the same as what
        would be written to a file.

        :return: String configuration block
        :rtype: str

        """
        fd, fp = tempfile.mkstemp()
        try:
            self.write(fp)
            with open(fp) as written_config:
                s = written_config.read()
        finally:
            os.close(fd)
            os.remove(fp)
        return s


def _initialize_cb_statics():
    """ Initialize ConfigBlock class variables from library """
    VitalObject.VITAL_LIB.vital_config_block_block_sep.restype = ctypes.c_char_p
    VitalObject.VITAL_LIB.vital_config_block_global_value.restype = ctypes.c_char_p
    ConfigBlock.BLOCK_SEP = VitalObject.VITAL_LIB.vital_config_block_block_sep()
    ConfigBlock.GLOBAL_VALUE = VitalObject.VITAL_LIB.vital_config_block_global_value()


# Only call if ConfigBlock globals are not initialized yet
if ConfigBlock.BLOCK_SEP is None:
    _initialize_cb_statics()
