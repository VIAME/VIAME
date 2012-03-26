/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "python_allow_threads.h"

/**
 * \file python_allow_threads.cxx
 *
 * \brief RAII class for calling into non-Python code.
 */

namespace vistk
{

namespace python
{

python_allow_threads
::python_allow_threads(bool save)
  : thread(save ? PyEval_SaveThread() : NULL)
{
}

python_allow_threads
::~python_allow_threads()
{
  release();
}

void
python_allow_threads
::release()
{
  if (thread)
  {
    PyEval_RestoreThread(thread);
    thread = NULL;
  }
}

}

}
