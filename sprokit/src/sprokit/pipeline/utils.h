// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
