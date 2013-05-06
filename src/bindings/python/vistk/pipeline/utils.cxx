/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <sprokit/pipeline/utils.h>

#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/module.hpp>

/**
 * \file utils.cxx
 *
 * \brief Python bindings for utils.
 */

using namespace boost::python;

BOOST_PYTHON_MODULE(utils)
{
  class_<sprokit::thread_name_t>("ThreadName"
    , "A type for the name of a thread.");

  def("name_thread", &sprokit::name_thread
    , (arg("name"))
    , "Names the currently running thread.");
}
