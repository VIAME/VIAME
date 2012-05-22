/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <python/helpers/python_convert_optional.h>

#include <vistk/utilities/timestamp.h>

#include <vistk/python/any_conversion/prototypes.h>
#include <vistk/python/any_conversion/registration.h>

#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/operators.hpp>

/**
 * \file timestamp.cxx
 *
 * \brief Python bindings for timestamp.
 */

using namespace boost::python;

BOOST_PYTHON_MODULE(timestamp)
{
  class_<vistk::timestamp>("Timestamp"
    , "A timestamp class."
    , no_init)
    .def(init<>())
    .def(init<vistk::timestamp::time_t>())
    .def(init<vistk::timestamp::frame_t>())
    .def(init<vistk::timestamp::time_t, vistk::timestamp::frame_t>())
    .def("has_time", &vistk::timestamp::has_time
      , "Returns True if the timestamp has a valid time, False otherwise.")
    .def("has_frame", &vistk::timestamp::has_frame
      , "Returns True if the timestamp has a valid frame, False otherwise.")
    .def("time", &vistk::timestamp::time
      , "The time of the timestamp.")
    .def("frame", &vistk::timestamp::frame
      , "The frame of the timestamp.")
    .def("set_time", &vistk::timestamp::set_time
      , "Set the time of the timestamp.")
    .def("set_frame", &vistk::timestamp::set_frame
      , "Set the frame of the timestamp.")
    .def("clear_time", &vistk::timestamp::clear_time
      , "Unset the time of the timestamp.")
    .def("clear_frame", &vistk::timestamp::clear_frame
      , "Unset the frame of the timestamp.")
    .def("is_valid", &vistk::timestamp::is_valid
      , "Returns True if the timestamp if valid, False otherwise.")
    .def(self == self)
    .def(self <  self)
    .def(self >  self)
  ;

  vistk::python::register_type<vistk::timestamp>(20);
}
