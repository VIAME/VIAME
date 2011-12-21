/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <python/helpers/python_wrap_const_shared_ptr.h>

#include <vistk/python/any_conversion/prototypes.h>
#include <vistk/python/any_conversion/registration.h>

#include <vistk/pipeline/datum.h>

#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/def.hpp>
#include <boost/python/enum.hpp>
#include <boost/python/exception_translator.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/implicit.hpp>
#include <boost/python/module.hpp>
#include <boost/python/object.hpp>
#include <boost/any.hpp>
#include <boost/cstdint.hpp>

#include <string>

/**
 * \file datum.cxx
 *
 * \brief Python bindings for \link vistk::datum\endlink.
 */

using namespace boost::python;

static void translator(vistk::datum_exception const& e);

static vistk::datum_t new_datum(object const& obj);

BOOST_PYTHON_MODULE(datum)
{
  register_exception_translator<
    vistk::datum_exception>(translator);

  enum_<vistk::datum::type_t>("DatumType"
    , "A type for a datum packet.")
    .value("invalid", vistk::datum::invalid)
    .value("data", vistk::datum::data)
    .value("empty", vistk::datum::empty)
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
  def("complete", &vistk::datum::complete_datum
    , "Creates a complete marker datum packet.");
  def("error", &vistk::datum::error_datum
    , (arg("err"))
    , "Creates an error datum packet.");

  class_<vistk::datum, vistk::datum_t, boost::noncopyable>("Datum"
    , "A packet of data within the pipeline."
    , no_init)
    .def("type", &vistk::datum::type
      , "The type of the datum packet.")
    .def("get_error", &vistk::datum::get_error
      , "The error contained within the datum packet.")
    .def("get_datum", &vistk::datum::get_datum<boost::any>
      , "Get the data contained within the packet.")
  ;

  implicitly_convertible<boost::shared_ptr<vistk::datum>, vistk::datum_t>();

  vistk::python::register_type<std::string>(0);
  vistk::python::register_type<int32_t>(1);
  vistk::python::register_type<char>(2);
  vistk::python::register_type<bool>(3);
  vistk::python::register_type<double>(4);

  implicitly_convertible<boost::any, object>();
  implicitly_convertible<object, boost::any>();
}

void
translator(vistk::datum_exception const& e)
{
  PyErr_SetString(PyExc_RuntimeError, e.what());
}

vistk::datum_t
new_datum(object const& obj)
{
  boost::any const any = extract<boost::any>(obj);

  return vistk::datum::new_datum(any);
}
