/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <sprokit/pipeline/datum.h>

#include <sprokit/python/any_conversion/prototypes.h>
#include <sprokit/python/any_conversion/registration.h>
#include <sprokit/python/util/python_gil.h>

#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/enum.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/implicit.hpp>
#include <boost/python/module.hpp>
#include <boost/python/object.hpp>
#include <boost/any.hpp>
#include <boost/cstdint.hpp>

#include <limits>
#include <string>

/**
 * \file datum.cxx
 *
 * \brief Python bindings for \link sprokit::datum\endlink.
 */

using namespace boost::python;

static sprokit::datum_t new_datum(object const& obj);
static sprokit::datum::type_t datum_type(sprokit::datum_t const& self);
static sprokit::datum::error_t datum_get_error(sprokit::datum_t const& self);
static object datum_get_datum(sprokit::datum_t const& self);

BOOST_PYTHON_MODULE(datum)
{
  enum_<sprokit::datum::type_t>("DatumType"
    , "A type for a datum packet.")
    .value("invalid", sprokit::datum::invalid)
    .value("data", sprokit::datum::data)
    .value("empty", sprokit::datum::empty)
    .value("flush", sprokit::datum::flush)
    .value("complete", sprokit::datum::complete)
    .value("error", sprokit::datum::error)
  ;

  class_<sprokit::datum::error_t>("DatumError"
    , "The type of an error message.");

  def("new", &new_datum
    , (arg("dat"))
    , "Creates a new datum packet.");
  def("empty", &sprokit::datum::empty_datum
    , "Creates an empty datum packet.");
  def("flush", &sprokit::datum::flush_datum
    , "Creates a flush marker datum packet.");
  def("complete", &sprokit::datum::complete_datum
    , "Creates a complete marker datum packet.");
  def("error", &sprokit::datum::error_datum
    , (arg("err"))
    , "Creates an error datum packet.");

  class_<sprokit::datum_t>("Datum"
    , "A packet of data within the pipeline."
    , no_init)
    .def("type", &datum_type
      , "The type of the datum packet.")
    .def("get_error", &datum_get_error
      , "The error contained within the datum packet.")
    .def("get_datum", &datum_get_datum
      , "Get the data contained within the packet.")
  ;

  sprokit::python::register_type<std::string>(0);
  sprokit::python::register_type<int32_t>(1);
  sprokit::python::register_type<char>(2);
  sprokit::python::register_type<bool>(3);
  sprokit::python::register_type<double>(4);

  // At worst, pass the object itself through.
  sprokit::python::register_type<object>(std::numeric_limits<sprokit::python::priority_t>::max());

  implicitly_convertible<boost::any, object>();
  implicitly_convertible<object, boost::any>();
}

sprokit::datum_t
new_datum(object const& obj)
{
  sprokit::python::python_gil const gil;

  (void)gil;

  boost::any const any = extract<boost::any>(obj)();

  return sprokit::datum::new_datum(any);
}

sprokit::datum::type_t
datum_type(sprokit::datum_t const& self)
{
  return self->type();
}

sprokit::datum::error_t
datum_get_error(sprokit::datum_t const& self)
{
  return self->get_error();
}

object
datum_get_datum(sprokit::datum_t const& self)
{
  sprokit::python::python_gil const gil;

  (void)gil;

  boost::any const any = self->get_datum<boost::any>();

  return object(any);
}
