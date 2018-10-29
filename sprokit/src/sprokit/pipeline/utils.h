/*ckwg +29
 * Copyright 2011-2018 by Kitware, Inc.
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

/**
 * \file utils.h
 *
 * \brief Common utilities when dealing with pipelines.
 */

#ifndef SPROKIT_PIPELINE_UTILS_H
#define SPROKIT_PIPELINE_UTILS_H

#include <sprokit/pipeline/sprokit_pipeline_export.h>

#include <string>
#include <typeinfo>

namespace sprokit {

/// The type for the name of a thread.
typedef std::string thread_name_t;

/**
 * \brief Name the thread that the function was called from.
 *
 * \note On Linux, the process name is limited to 16 characters by default.
 * Recompiling the kernel can expand this space. The \p name given is truncated
 * by the kernel automatically.
 *
 * \note On Windows, this function only succeeds if \c NDEBUG is not defined.
 * This is to help performance.
 *
 * \param name The name of the thread.
 *
 * \returns True if the name was successfully set, false otherwise.
 */
SPROKIT_PIPELINE_EXPORT bool name_thread(thread_name_t const& name);

} // end namespace

#endif // SPROKIT_PIPELINE_UTILS_H
