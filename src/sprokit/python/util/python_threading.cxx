/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "python_threading.h"

#include <Python.h>

namespace sprokit
{

namespace python
{

python_threading
::python_threading()
{
  if (!PyEval_ThreadsInitialized())
  {
    PyEval_InitThreads();
  }
}

python_threading
::~python_threading()
{
}

}

}
