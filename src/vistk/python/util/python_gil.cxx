/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "python_gil.h"

namespace sprokit
{

namespace python
{

python_gil
::python_gil()
  : state(PyGILState_Ensure())
{
}

python_gil
::~python_gil()
{
  PyGILState_Release(state);
}

}

}
