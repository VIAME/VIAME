// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/vital/util/python.h>
#include <python/kwiver/vital/util/python_exceptions.h>

namespace kwiver {
namespace vital {
namespace python {

void
python_print_exception()
{
  PyObject* type;
  PyObject* value;
  PyObject* traceback;

  // Increments refcounts for returns.
  PyErr_Fetch(&type, &value, &traceback);

  // Increment ourselves.
  Py_XINCREF(type);
  Py_XINCREF(value);
  Py_XINCREF(traceback);

  // Put the error back (decrements refcounts).
  PyErr_Restore(type, value, traceback);

  // Print the error (also clears it).
  PyErr_PrintEx(0);

  // Put the error back for everyone else.
  PyErr_Restore(type, value, traceback);
}

} } } // end of namespaces
