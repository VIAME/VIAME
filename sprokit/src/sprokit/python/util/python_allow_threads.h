/*ckwg +29
 * Copyright 2012 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SPROKIT_PYTHON_UTIL_PYTHON_ALLOW_THREADS_H
#define SPROKIT_PYTHON_UTIL_PYTHON_ALLOW_THREADS_H

#include "util-config.h"

#include <boost/noncopyable.hpp>

#include <sprokit/python/util/python.h>

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
