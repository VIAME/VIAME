/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PYTHON_UTIL_PYTHON_GIL_H
#define VISTK_PYTHON_UTIL_PYTHON_GIL_H

#include "util-config.h"

#include <boost/noncopyable.hpp>

#include <Python.h>

/**
 * \file python_gil.h
 *
 * \brief RAII class for grabbing the Python GIL.
 */

namespace vistk
{

namespace python
{

/**
 * \class python_gil python_gil.h <vistk/python/util/python_gil.h>
 *
 * \brief Grabs the Python GIL and uses RAII to ensure it is released.
 */
class VISTK_PYTHON_UTIL_EXPORT python_gil
  : public boost::noncopyable
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

#endif // VISTK_PYTHON_UTIL_PYTHON_GIL_H
