/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PYTHON_UTIL_PYTHON_ALLOW_THREADS_H
#define VISTK_PYTHON_UTIL_PYTHON_ALLOW_THREADS_H

#include "util-config.h"

#include <boost/noncopyable.hpp>

#include <Python.h>

/**
 * \file python_allow_threads.h
 *
 * \brief RAII class for calling into non-Python code.
 */

namespace vistk
{

namespace python
{

/**
 * \class python_allow_threads python_allow_threads.h <vistk/python/util/python_allow_threads.h>
 *
 * \brief RAII class for calling into non-Python code.
 */
class VISTK_PYTHON_UTIL_EXPORT python_allow_threads
  : boost::noncopyable
{
  public:
    /**
     * \brief Constructor.
     *
     * \param save If \c true, saves the state; is a no-op if \c false.
     */
    python_allow_threads(bool save = true);
    /**
     * \brief Destructor.
     */
    ~python_allow_threads();

    /**
     * \brief Manually acquire the GIL again.
     */
    void release();
  private:
    PyThreadState* thread;
};

}

}

#endif // VISTK_PYTHON_UTIL_PYTHON_ALLOW_THREADS_H
