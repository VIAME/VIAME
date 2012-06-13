/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline/modules.h>

#include <boost/python/def.hpp>
#include <boost/python/module.hpp>

/**
 * \file modules.cxx
 *
 * \brief Python bindings for module loading.
 */

using namespace boost::python;

BOOST_PYTHON_MODULE(modules)
{
  def("load_known_modules", &vistk::load_known_modules
    , "Loads vistk modules to populate the process and scheduler registries.");
}
