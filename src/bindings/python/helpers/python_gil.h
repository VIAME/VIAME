/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PYTHON_HELPERS_PYTHON_GIL_H
#define VISTK_PYTHON_HELPERS_PYTHON_GIL_H

#include <Python.h>

class python_gil
{
  public:
    python_gil();
    ~python_gil();
  private:
    PyGILState_STATE const state;
};

#endif // VISTK_PYTHON_HELPERS_PYTHON_GIL_H
