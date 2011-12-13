/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <python/helpers/python_wrap_const_shared_ptr.h>

#include <vistk/pipeline/datum.h>

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

/**
 * \file datum.cxx
 *
 * \brief Python bindings for \link vistk::datum\endlink.
 */

using namespace boost::python;

static void translator(vistk::datum_exception const& e);

BOOST_PYTHON_MODULE(datum)
{
  register_exception_translator<
    vistk::datum_exception>(translator);

  enum_<vistk::datum::datum_type_t>("DatumType")
    .value("INVALID", vistk::datum::DATUM_INVALID)
    .value("DATA", vistk::datum::DATUM_DATA)
    .value("EMPTY", vistk::datum::DATUM_EMPTY)
    .value("COMPLETE", vistk::datum::DATUM_COMPLETE)
    .value("ERROR", vistk::datum::DATUM_ERROR)
  ;

  class_<vistk::datum::error_t>("DatumError");

  /// \todo How to do this?
  //def("new", &vistk::datum::new_datum);
  def("empty", &vistk::datum::empty_datum);
  def("complete", &vistk::datum::complete_datum);
  def("error", &vistk::datum::error_datum);

  class_<vistk::datum, vistk::datum_t, boost::noncopyable>("Datum", no_init)
    .def("type", &vistk::datum::type)
    .def("get_error", &vistk::datum::get_error)
    /// \todo How to do this?
    //.def("get_datum", &vistk::datum::get_value)
  ;
}

void
translator(vistk::datum_exception const& e)
{
  PyErr_SetString(PyExc_RuntimeError, e.what());
}
