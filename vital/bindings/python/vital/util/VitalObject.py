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

Base class for all VITAL Python interface classes

"""

import abc
import ctypes
import logging

from vital.util.find_vital_library import find_vital_library


class VitalClassMeta (abc.ABCMeta):
    """
    Metaclass for Vital object types.

    Ensures that C_TYPE and C_TYPE_PTR are defined in derived classes.
    """

    def __new__(cls, name, bases, attrs):

        # Create a new structure type for the class if it has not already
        # defined one for itself
        if 'C_TYPE' not in attrs:
            class OpaqueStruct (ctypes.Structure):
                pass
            OpaqueStruct.__name__ = "%sOpaqueStruct" % name
            attrs['C_TYPE'] = OpaqueStruct
            attrs['C_TYPE_PTR'] = ctypes.POINTER(OpaqueStruct)

        return super(VitalClassMeta, cls).__new__(cls, name, bases, attrs)


class VitalObject (object):
    """
    Basic VITAL python interface class.

    Guarantees that should be maintained:
        - c_type() and c_ptr_type() should be used when trying to get C types
          from class types.
        - C_TYPE and C_TYPE_PTR should be used when trying to get C types from
          class instances, and thus should only refer to a single type in an
          instance. Value undefined defined on the class level.

    """
    __metaclass__ = VitalClassMeta

    VITAL_LIB = find_vital_library()

    # C API opaque structure + pointer
    C_TYPE = None
    C_TYPE_PTR = None

    @classmethod
    def c_type(cls, *args):
        """ Get the C opaque type """
        return cls.C_TYPE

    @classmethod
    def c_ptr_type(cls, *args):
        """ Get the C opaque pointer type """
        return cls.C_TYPE_PTR

    @classmethod
    def _call_cfunc(cls, func_name, argtypes, restype, *args):
        """
        Extract function from vital library and call it with a VitalErrorHandle.

        This assumes that the C function takes an additional parameter than what
        is given to this function that is the error handle.

        :param func_name: C function name to pull from library
        :type func_name: str

        :param argtypes: Ctypes argument type array
        :type argtypes: list | tuple

        :param restype: Ctypes return type

        :param args: iterable of positional arguments to the C function
        :param args: tuple

        :return: Result of the c function call

        """
        # local import to prevent circular import
        from vital.util import VitalErrorHandle
        f = cls.VITAL_LIB[func_name]
        if argtypes:
            f.argtypes = list(argtypes) + [VitalErrorHandle.c_ptr_type()]
        f.restype = restype
        with VitalErrorHandle() as eh:
            return f(*(args + (eh,)))

    def __init__(self, from_cptr=None, *args, **kwds):
        """
        Create a new instance of the Python vital type wrapper.

        This initializer should only be called after C_TYPE/C_TYPE_PTR are
        concrete types.

        :param from_cptr: Existing C opaque instance pointer to use, preventing
            new instance construction. This should of course be a valid pointer
            to an instance. Only a new instance pointer or a new shared pointer
            reference should be passed here, otherwise memory issue will ensue.
            Thus this should only be used if you know what you're doing.

        Optional keyword arguments:

        :param allow_null_pointer: Allow a null pointer to be returned from the
            _new method instead of raising an exception.

        """
        self._inst_ptr = None

        if None in (self.C_TYPE, self.C_TYPE_PTR):
            raise RuntimeError("Derived class did not define opaque handle "
                               "structure types.")

        allow_null_pointer = kwds.get('allow_null_pointer', None)
        if allow_null_pointer is not None:
            del kwds['allow_null_pointer']

        if from_cptr is not None:
            # if null pointer and we're not allowing them
            if not (allow_null_pointer or bool(from_cptr)):
                raise RuntimeError("Cannot initialize to a null pointer")
            # if not a valid opaque pointer type
            elif not isinstance(from_cptr, self.C_TYPE_PTR):
                raise RuntimeError("Given C Opaque Pointer is not of the "
                                   "correct type. Given '%s' but expected '%s'."
                                   % (type(from_cptr), self.C_TYPE_PTR))
            self._inst_ptr = from_cptr
        else:
            self._inst_ptr = self._new(*args, **kwds)
            # raise if we have a null pointer and we don't allow nulls
            if not (allow_null_pointer or bool(self._inst_ptr)):
                raise RuntimeError("Failed to construct new %s instance: Null "
                                   "pointer returned from construction."
                                   % self.__class__.__name__)

    def __del__(self):
        if hasattr(self, '_inst_ptr') and self._inst_ptr is not None:
            self._destroy()

    def __nonzero__(self):
        """ bool() operator for 2.x """
        return bool(self.c_pointer)

    def __bool__(self):
        """ bool() operator for 3.x """
        return bool(self.c_pointer)

    @property
    def _as_parameter_(self):
        """
        Ctypes interface attribute for allowing a user to pass the python object
        instance as argument to a C function instead of the opaque pointer.
        This means that when an instance of this class is passed as an argument,
        the underlying opaque pointer is automatically passed in its place.
        """
        return self.c_pointer

    @classmethod
    def logger(cls):
        return logging.getLogger('.'.join([cls.__module__, cls.__name__]))

    @property
    def _log(self):
        return self.logger()

    @property
    def c_pointer(self):
        """
        :return: The ctypes opaque structure pointer
        :rtype: _ctypes._Pointer
        """
        return self._inst_ptr

    @abc.abstractmethod
    def _new(self, *args, **kwds):
        """
        Construct a new instance, returning new instance opaque C pointer and
        initializing any other necessary object properties

        :returns: New C opaque structure pointer.
        :rtype: _ctypes._Pointer

        """
        raise NotImplementedError("Calling VitalObject class abstract _new "
                                  "method.")

    @abc.abstractmethod
    def _destroy(self):
        """
        Call C API destructor for derived class
        """
        raise NotImplementedError("Calling VitalObject class abstract _destroy "
                                  "method.")

    # TODO: Serialization hooks?


class OpaqueTypeCache (object):
    """
    Support structure for VitalObject sub-classes that represent multiple
    C types akin to C++ templating.
    """

    def __init__(self, name_prefix=None):
        # Store pairs of C opaque structure and its pointer type
        #: :type: dict[str, (_ctypes.PyCStructType, _ctypes.PyCPointerType)]
        self._c_type_cache = {}
        self._prefix = name_prefix or ''

    def get_types(self, k):
        """
        Return or generate opaque type and pointer based on shape spec
        """
        if k not in self._c_type_cache:
            # Based on VitalClassMetadata meta-cass
            class OpaqueStruct (ctypes.Structure):
                pass
            OpaqueStruct.__name__ = "%s%s_OpaqueStructure" % (self._prefix, k)
            self._c_type_cache[k] = \
                (OpaqueStruct, ctypes.POINTER(OpaqueStruct))
        return self._c_type_cache[k]

    def new_type_getter(self):
        """
        Returns new simple object with a __getitem__ hook for getting a specific
        C opaque type.
        """
        class c_type_manager (object):
            def __getitem__(s2, k):
                return self.get_types(k)[0]

            __contains__ = self._c_type_cache.__contains__

            @property
            def _as_parameter_(self):
                raise RuntimeError("Cannot use type manager as ctypes "
                                   "parameter")

        return c_type_manager()

    def new_ptr_getter(self):
        """
        Returns new simple object with a __getitem__ hook for getting a specific
        C opaque pointer type.
        """
        class c_type_ptr_manager (object):
            def __getitem__(s2, k):
                return self.get_types(k)[1]

            __contains__ = self._c_type_cache.__contains__

            @property
            def _as_parameter_(self):
                raise RuntimeError("Cannot use type manager as ctypes "
                                   "parameter")

        return c_type_ptr_manager()
