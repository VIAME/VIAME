/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef SPROKIT_PYTHON_UTIL_PYTHON_GIL_H
#define SPROKIT_PYTHON_UTIL_PYTHON_GIL_H

#include "util-config.h"

#include <boost/noncopyable.hpp>

#include <Python.h>

/**
 * \file python_gil.h
 *
 * \brief RAII class for grabbing the Python GIL.
 */

namespace sprokit
{

namespace python
{

/**
 * \class python_gil python_gil.h <sprokit/python/util/python_gil.h>
 *
 * \brief Grabs the Python GIL and uses RAII to ensure it is released.
 */
class SPROKIT_PYTHON_UTIL_EXPORT python_gil
  : boost::noncopyable
{
  public:
    /**
     * \brief Constructor.
     */
    python_gil();
    /**
     * \brief Destructor.
     */
    ~python_gil();
  private:
    PyGILState_STATE const state;
};

}

}

#endif // SPROKIT_PYTHON_UTIL_PYTHON_GIL_H
