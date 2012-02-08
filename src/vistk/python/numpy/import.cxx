/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "import.h"

#include <boost/thread/once.hpp>

#include <Python.h>

#include <numpy/arrayobject.h>

namespace vistk
{

namespace python
{

static void import_numpy_module();

void
import_numpy()
{
  static boost::once_flag once;

  boost::call_once(once, import_numpy_module);
}

void
import_numpy_module()
{
  import_array();
}

}

}
