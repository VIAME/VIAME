/*ckwg +29
 * Copyright 2011-2015 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#if WIN32
#pragma warning (push)
#pragma warning (disable : 4244)
#endif
#include <pybind11/pybind11.h>
#include <vital/any.h>
#if WIN32
#pragma warning (pop)
#endif

#include <sprokit/pipeline/datum.h>

#include <sprokit/python/util/python_gil.h>

#include <limits>
#include <string>
#include <cstdint>

/**
 * \file datum.cxx
 *
 * \brief Python bindings for \link sprokit::datum\endlink.
 */

using namespace pybind11;

static sprokit::datum new_datum(object const& obj);
static sprokit::datum new_int_datum(object const& obj);
static sprokit::datum new_float_datum(object const& obj);
static sprokit::datum new_string_datum(object const& obj);
static sprokit::datum empty_datum();
static sprokit::datum flush_datum();
static sprokit::datum complete_datum();
static sprokit::datum error_datum(std::string const& err);
static sprokit::datum::type_t datum_type(sprokit::datum const& self);
static sprokit::datum::error_t datum_get_error(sprokit::datum const& self);
static object datum_get_datum(sprokit::datum const& self);
static std::string datum_datum_type(sprokit::datum const& self);

static PyObject* datum_get_datum_ptr(sprokit::datum& self);
static sprokit::datum_t datum_from_capsule( PyObject* cap );

char const* sprokit_datum_PyCapsule_name() { return  "sprokit::datum"; }


PYBIND11_MODULE(datum, m)
{

  enum_<sprokit::datum::type_t>(m, "DatumType"
    , "A type for a datum packet.")
    .value("invalid", sprokit::datum::invalid)
    .value("data", sprokit::datum::data)
    .value("empty", sprokit::datum::empty)
    .value("flush", sprokit::datum::flush)
    .value("complete", sprokit::datum::complete)
    .value("error", sprokit::datum::error)
  ;

  // constructors
  m.def("new", &new_datum
    , (arg("dat"))
    , "Creates a new datum packet containing a python object.");
  m.def("new_int", &new_int_datum
    , (arg("dat"))
    , "Creates a new datum packet containing an int.");
  m.def("new_float", &new_float_datum
    , (arg("dat"))
    , "Creates a new datum packet containing a float.");
  m.def("new_string", &new_string_datum
    , (arg("dat"))
    , "Creates a new datum packet containing a string.");
  m.def("datum_from_capsule", &datum_from_capsule
      , (arg("dptr"))
      , "Converts datum* in capsule to datum_t");
  m.def("empty", &empty_datum
    , "Creates an empty datum packet.");
  m.def("flush", &flush_datum
    , "Creates a flush marker datum packet.");
  m.def("complete", &complete_datum
    , "Creates a complete marker datum packet.");
  m.def("error", &error_datum
    , arg("err")
    , "Creates an error datum packet.");

  // Methods on datum
  class_<sprokit::datum>(m, "Datum"
    , "A packet of data within the pipeline.")
    .def("type", &datum_type
      , "The type of the datum packet.")
    .def("datum_type", &datum_datum_type
      , "The type of the data in the packet")
    .def("get_error", &datum_get_error
      , "The error contained within the datum packet.")
    .def("get_datum", &datum_get_datum
      , "Get the data contained within the packet.")
    .def("get_datum_ptr", &datum_get_datum_ptr
      , "Get pointer to datum object as a PyCapsule.")
  ;

} // end module


// ------------------------------------------------------------------

// For now, we need to manually specify how we want to cast our datum
// This should be fixed when we move away from kwiver::vital::any
sprokit::datum
new_datum(object const& obj)
{
  return *(sprokit::datum::new_datum(obj));
}

sprokit::datum
new_int_datum(object const& obj)
{
  return *(sprokit::datum::new_datum(cast<int>(obj)));
}

sprokit::datum
new_float_datum(object const& obj)
{
  return *(sprokit::datum::new_datum(cast<float>(obj)));
}

sprokit::datum
new_string_datum(object const& obj)
{
  return *(sprokit::datum::new_datum(cast<std::string>(obj)));
}

sprokit::datum
empty_datum()
{
  return *(sprokit::datum::empty_datum());
}

sprokit::datum
flush_datum()
{
  return *(sprokit::datum::flush_datum());
}

sprokit::datum
complete_datum()
{
  return *(sprokit::datum::complete_datum());
}

sprokit::datum
error_datum(std::string const& err)
{
  return *(sprokit::datum::error_datum(err));
}

sprokit::datum::type_t
datum_type(sprokit::datum const& self)
{
  return self.type();
}

sprokit::datum::error_t
datum_get_error(sprokit::datum const& self)
{
  return self.get_error();
}

object
datum_get_datum(sprokit::datum const& self)
{
  object dat = none();
  if ( self.type() == sprokit::datum::data )
  {
    kwiver::vital::any const any = self.get_datum<kwiver::vital::any>();
    dat = kwiver::vital::any_cast<object>(any);
  }

  return dat;
}

std::string
datum_datum_type(sprokit::datum const& self)
{
  kwiver::vital::any const any = self.get_datum<kwiver::vital::any>();

  return any.type().name();
}

// ------------------------------------------------------------------
// Bridge regular python to pybind11

/**
 * \brief Get address of datum object.
 *
 * This function returns the address of the datum object managed by
 * the datum_t sptr.
 *
 * Caller holds the datum_t while we are returning the address of the
 * datum.  The customer then extracts the data contained in the datum.
 * After this the datum is then expendable.
 *
 * \param self Reference to datum_t (sptr)
 *
 * \return Address of real datum object.
 */
PyObject*
datum_get_datum_ptr(sprokit::datum& self)
{
  return PyCapsule_New( const_cast< sprokit::datum* >(std::make_shared<sprokit::datum> (self).get()), "sprokit::datum", NULL );
}


/**
 * \brief Convert PyCapsule to datum_t sptr.
 *
 * This function converts the PyCapsule containing the address of a
 * datum object into a datum_t sptr that is managing that object.
 *
 * \param cap Pointer to PyCapsule that contains address of datum object.
 *
 * \return datum_t sptr that manages supplied datum object.
 */
sprokit::datum_t
datum_from_capsule( PyObject* cap )
{
  // cap is pointer to datum
  if (PyCapsule_IsValid( cap, "sprokit::datum" ))
  {
    sprokit::datum* dptr = static_cast<sprokit::datum*>( PyCapsule_GetPointer( cap, "sprokit::datum" ) );
    return sprokit::datum_t(dptr);
  }

  return sprokit::datum::error_datum( "Invalid PyCapsule" );
}
