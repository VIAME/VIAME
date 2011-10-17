/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <python/helpers/python_wrap_const_shared_ptr.h>
#include <python/helpers/python_convert_any.h>

#include <vistk/pipeline/datum.h>

#include <boost/any.hpp>
#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

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
  to_python_converter<boost::any, boost_any_to_object>();

  boost_any_to_object();
}

void
translator(vistk::datum_exception const& e)
{
  PyErr_SetString(PyExc_RuntimeError, e.what());
}

vistk::datum_t
new_datum(object const& obj)
{
  return vistk::datum::new_datum<boost::any>(extract<boost::any>(obj));
}
