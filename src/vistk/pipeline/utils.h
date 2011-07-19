/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_UTILS_H
#define VISTK_PIPELINE_UTILS_H

#include "pipeline-config.h"

#include <string>

/**
 * \file utils.h
 *
 * \brief Common utilities when dealing with pipelines.
 */

namespace vistk
{

/// The type for the name of a thread.
typedef std::string thread_name_t;

/**
 * \brief Names the thread that the function was called from.
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
bool VISTK_PIPELINE_EXPORT name_thread(thread_name_t const& name);

}

#endif // VISTK_PIPELINE_UTILS_H
