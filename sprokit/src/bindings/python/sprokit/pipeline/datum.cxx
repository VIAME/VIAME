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
#include <pybind11/stl_bind.h>
#include <vital/any.h>
#if WIN32
#pragma warning (pop)
#endif

#include <sprokit/pipeline/datum.h>

// Type conversions
#include <vital/types/image_container.h>
#include <vital/types/detected_object_set.h>
#include <vital/types/descriptor_set.h>
#include <vital/types/track_set.h>
#include <vital/types/object_track_set.h>
#include <vital/types/feature_track_set.h>
#include <vital/types/timestamp.h>

#include <limits>
#include <string>
#include <cstdint>

/**
 * \file datum.cxx
 *
 * \brief Python bindings for \link sprokit::datum\endlink.
 */

using namespace pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<unsigned char>);
PYBIND11_MAKE_OPAQUE(std::vector<double>);
PYBIND11_MAKE_OPAQUE(std::vector<std::string>);

static sprokit::datum new_datum_no_cast(object const& obj);
template<class T> sprokit::datum new_datum(object const& obj);
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

template<class T> T datum_get_object(sprokit::datum &);

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

  bind_vector<std::vector<unsigned char>, std::shared_ptr<std::vector<unsigned char>>>(m, "VectorUChar");
  bind_vector<std::vector<double>, std::shared_ptr<std::vector<double>>>(m, "VectorDouble");
  bind_vector<std::vector<std::string>, std::shared_ptr<std::vector<std::string>>>(m, "VectorString");

  // constructors
  m.def("new", &new_datum_no_cast
    , (arg("dat"))
    , "Creates a new datum packet containing a python object.");
  m.def("new_int", &new_datum<int>
    , (arg("dat"))
    , "Creates a new datum packet containing an int.");
  m.def("new_float", &new_datum<float>
    , (arg("dat"))
    , "Creates a new datum packet containing a float.");
  m.def("new_string", &new_datum<std::string>
    , (arg("dat"))
    , "Creates a new datum packet containing a string.");
  m.def("new_image_container", &new_datum<std::shared_ptr<kwiver::vital::image_container>>
    , (arg("dat"))
    , "Creates a new datum packet containing an image container.");
  m.def("new_descriptor_set", &new_datum<std::shared_ptr<kwiver::vital::descriptor_set>>
    , (arg("dat"))
    , "Creates a new datum packet containing a descriptor set.");
  m.def("new_detected_object_set", &new_datum<std::shared_ptr<kwiver::vital::detected_object_set>>
    , (arg("dat"))
    , "Creates a new datum packet containing a detected object set.");
  m.def("new_track_set", &new_datum<std::shared_ptr<kwiver::vital::track_set>>
    , (arg("dat"))
    , "Creates a new datum packet containing a track set.");
  m.def("new_feature_track_set", &new_datum<std::shared_ptr<kwiver::vital::feature_track_set>>
    , (arg("dat"))
    , "Creates a new datum packet containing a feature track set.");
  m.def("new_object_track_set", &new_datum<std::shared_ptr<kwiver::vital::object_track_set>>
    , (arg("dat"))
    , "Creates a new datum packet containing an object track set.");
  m.def("new_timestamp", &new_datum<std::shared_ptr<kwiver::vital::timestamp>>
    , (arg("dat"))
    , "Creates a new datum packet containing a timestamp.");
  m.def("new_double_vector", &new_datum<std::shared_ptr<std::vector<double>>>
    , (arg("dat"))
    , "Creates a new datum packet containing a double vector.");
  m.def("new_string_vector", &new_datum<std::shared_ptr<std::vector<std::string>>>
    , (arg("dat"))
    , "Creates a new datum packet containing a string vector.");
  m.def("new_uchar_vector", &new_datum<std::shared_ptr<std::vector<unsigned char>>>
    , (arg("dat"))
    , "Creates a new datum packet containing an unsigned char vector.");
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
      , "Get the data contained within the packet (if coming from a python process).")
    .def("get_datum_ptr", &datum_get_datum_ptr
      , "Get pointer to datum object as a PyCapsule.")
    .def("get_string", &datum_get_object<std::string>
      , "Convert the data to a string")
    .def("get_image_container", &datum_get_object<std::shared_ptr<kwiver::vital::image_container>>
      , "Convert the data to an image container")
    .def("get_descriptor_set", &datum_get_object<std::shared_ptr<kwiver::vital::descriptor_set>>
      , "Convert the data to a descriptor set")
    .def("get_detected_object_set", &datum_get_object<std::shared_ptr<kwiver::vital::detected_object_set>>
      , "Convert the data to a detected object set")
    .def("get_track_set", &datum_get_object<std::shared_ptr<kwiver::vital::track_set>>
      , "Convert the data to a track set")
    .def("get_feature_track_set", &datum_get_object<std::shared_ptr<kwiver::vital::feature_track_set>>
      , "Convert the data to a feature track set")
    .def("get_object_track_set", &datum_get_object<std::shared_ptr<kwiver::vital::object_track_set>>
      , "Convert the data to an object track set")
    .def("get_timestamp", &datum_get_object<std::shared_ptr<kwiver::vital::timestamp>>
      , "Convert the data to a timestamp")
    .def("get_double_vector", &datum_get_object<std::shared_ptr<std::vector<double>>>
      , "Convert the data to a double vector")
    .def("get_string_vector", &datum_get_object<std::shared_ptr<std::vector<std::string>>>
      , "Convert the data to a string vector")
    .def("get_uchar_vector", &datum_get_object<std::shared_ptr<std::vector<unsigned char>>>
      , "Convert the data to an unsigned char vector")
  ;

} // end module


// ------------------------------------------------------------------

// For now, we need to manually specify how we want to cast our datum
// This should be fixed when we move away from kwiver::vital::any
sprokit::datum
new_datum_no_cast(object const& obj)
{
  return *(sprokit::datum::new_datum(obj));
}

template<class T>
sprokit::datum
new_datum(object const& obj)
{
  return *(sprokit::datum::new_datum(cast<T>(obj)));
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

// This converts straight to a pybind11::object
// It can be useful if we can assume we're dealing specifically with python
// otherwise, use a converter like get_image_container
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

// ------------------------------------------------------------------
// Converter

/**
 * \brief Converter from python datum object to a value
 *
 * This function allows python to choose the type to convert a datum to.
 * The type, T, must have a pybind11 binding.
 * To use this, add a binding with an explicit type.
 *
 * \return shared_ptr<T> to the converted object
 */
template<class T>
T
datum_get_object(sprokit::datum &self)
{
  kwiver::vital::any const any = self.get_datum<kwiver::vital::any>();
  return kwiver::vital::any_cast<T>(any);
}
