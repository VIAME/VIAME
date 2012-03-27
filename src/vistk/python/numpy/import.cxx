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

namespace
{

typedef
#if PY_VERSION_HEX >= 0x03000000
  PyObject*
#else
  void
#endif
  pyimport_return_t;

}

static pyimport_return_t import_numpy_module();

void
import_numpy()
{
  static boost::once_flag once;

  boost::call_once(once, import_numpy_module);
}

pyimport_return_t
import_numpy_module()
{
  import_array();

#if PY_VERSION_HEX >= 0x03000000
  return NULL;
#endif
}

}

}
