/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <python/helpers/python_wrap_const_shared_ptr.h>

#include <vistk/pipeline/stamp.h>

#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/implicit.hpp>
#include <boost/python/module.hpp>
#include <boost/python/operators.hpp>
#include <boost/python/self.hpp>

/**
 * \file stamp.cxx
 *
 * \brief Python bindings for \link vistk::stamp\endlink.
 */

using namespace boost::python;

BOOST_PYTHON_MODULE(stamp)
{
  def("new_stamp", &vistk::stamp::new_stamp
    , "Creates a new stamp.");
  def("copied_stamp", &vistk::stamp::copied_stamp
    , (arg("stamp"))
    , "Creates an equivalent stamp to the one given.");
  def("incremented_stamp", &vistk::stamp::incremented_stamp
    , (arg("stamp"))
    , "Creates a stamp that is greater than the given stamp.");
  def("recolored_stamp", &vistk::stamp::recolored_stamp
    , (arg("stamp"), arg("colored_stamp"))
    , "Creates a copy of the given stamp with a new color.");

  class_<vistk::stamp, vistk::stamp_t, boost::noncopyable>("Stamp"
    , "An identifier to help synchronize data within the pipeline."
    , no_init)
    .def("is_same_color", &vistk::stamp::is_same_color
      , (arg("stamp"))
      , "Returns True if the stamps are the same color, False otherwise.")
    .def(self == self)
    .def(self < self)
  ;

  implicitly_convertible<boost::shared_ptr<vistk::stamp>, vistk::stamp_t>();
}
