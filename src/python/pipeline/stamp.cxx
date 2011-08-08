/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <python/helpers/python_wrap_const_shared_ptr.h>

#include <vistk/pipeline/stamp.h>

#include <boost/python.hpp>

/**
 * \file stamp.cxx
 *
 * \brief Python bindings for \link vistk::stamp\endlink.
 */

using namespace boost::python;

BOOST_PYTHON_MODULE(stamp)
{
  def("new_stamp", &vistk::stamp::new_stamp);
  def("copied_stamp", &vistk::stamp::copied_stamp);
  def("incremented_stamp", &vistk::stamp::incremented_stamp);
  def("recolored_stamp", &vistk::stamp::recolored_stamp);

  class_<vistk::stamp, vistk::stamp_t, boost::noncopyable>("Stamp", no_init)
    .def("is_same_color", &vistk::stamp::is_same_color)
    .def(self == self)
    .def(self < self)
  ;
}
