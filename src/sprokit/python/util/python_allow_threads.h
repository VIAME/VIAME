/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef SPROKIT_PYTHON_UTIL_PYTHON_ALLOW_THREADS_H
#define SPROKIT_PYTHON_UTIL_PYTHON_ALLOW_THREADS_H

#include "util-config.h"

#include <boost/noncopyable.hpp>

#include <Python.h>

/**
 * \file python_allow_threads.h
 *
 * \brief RAII class for calling into non-Python code.
 */

namespace sprokit
{

namespace python
{

/**
 * \class python_allow_threads python_allow_threads.h <sprokit/python/util/python_allow_threads.h>
 *
 * \brief RAII class for calling into non-Python code.
 */
class SPROKIT_PYTHON_UTIL_EXPORT python_allow_threads
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

#endif // SPROKIT_PYTHON_UTIL_PYTHON_ALLOW_THREADS_H
