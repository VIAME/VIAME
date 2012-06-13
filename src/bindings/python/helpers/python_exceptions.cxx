/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "python_exceptions.h"

#include <Python.h>

void
python_print_exception()
{
  PyObject* type;
  PyObject* value;
  PyObject* traceback;

  PyErr_Fetch(&type, &value, &traceback);

  Py_XINCREF(type);
  Py_XINCREF(value);
  Py_XINCREF(traceback);

  PyErr_Restore(type, value, traceback);

  PyErr_PrintEx(0);

  PyErr_Restore(type, value, traceback);
}
