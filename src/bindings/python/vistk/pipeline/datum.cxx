/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline/datum.h>

#include <vistk/python/any_conversion/prototypes.h>
#include <vistk/python/any_conversion/registration.h>
#include <vistk/python/util/python_gil.h>

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
 * \brief Python bindings for \link vistk::datum\endlink.
 */

using namespace boost::python;

static vistk::datum_t new_datum(object const& obj);
static vistk::datum::type_t datum_type(vistk::datum_t const& self);
static vistk::datum::error_t datum_get_error(vistk::datum_t const& self);
static object datum_get_datum(vistk::datum_t const& self);

BOOST_PYTHON_MODULE(datum)
{
  enum_<vistk::datum::type_t>("DatumType"
    , "A type for a datum packet.")
    .value("invalid", vistk::datum::invalid)
    .value("data", vistk::datum::data)
    .value("empty", vistk::datum::empty)
    .value("flush", vistk::datum::flush)
    .value("complete", vistk::datum::complete)
    .value("error", vistk::datum::error)
  ;

  class_<vistk::datum::error_t>("DatumError"
    , "The type of an error message.");

  def("new", &new_datum
    , (arg("dat"))
    , "Creates a new datum packet.");
  def("empty", &vistk::datum::empty_datum
    , "Creates an empty datum packet.");
  def("flush", &vistk::datum::flush_datum
    , "Creates a flush marker datum packet.");
  def("complete", &vistk::datum::complete_datum
    , "Creates a complete marker datum packet.");
  def("error", &vistk::datum::error_datum
    , (arg("err"))
    , "Creates an error datum packet.");

  class_<vistk::datum_t>("Datum"
    , "A packet of data within the pipeline."
    , no_init)
    .def("type", &datum_type
      , "The type of the datum packet.")
    .def("get_error", &datum_get_error
      , "The error contained within the datum packet.")
    .def("get_datum", &datum_get_datum
      , "Get the data contained within the packet.")
  ;

  vistk::python::register_type<std::string>(0);
  vistk::python::register_type<int32_t>(1);
  vistk::python::register_type<char>(2);
  vistk::python::register_type<bool>(3);
  vistk::python::register_type<double>(4);

  // At worst, pass the object itself through.
  vistk::python::register_type<object>(std::numeric_limits<vistk::python::priority_t>::max());

  implicitly_convertible<boost::any, object>();
  implicitly_convertible<object, boost::any>();
}

vistk::datum_t
new_datum(object const& obj)
{
  vistk::python::python_gil const gil;

  (void)gil;

  boost::any const any = extract<boost::any>(obj)();

  return vistk::datum::new_datum(any);
}

vistk::datum::type_t
datum_type(vistk::datum_t const& self)
{
  return self->type();
}

vistk::datum::error_t
datum_get_error(vistk::datum_t const& self)
{
  return self->get_error();
}

object
datum_get_datum(vistk::datum_t const& self)
{
  vistk::python::python_gil const gil;

  (void)gil;

  boost::any const any = self->get_datum<boost::any>();

  return object(any);
}
