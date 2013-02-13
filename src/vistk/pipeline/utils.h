/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_UTILS_H
#define VISTK_PIPELINE_UTILS_H

#include "pipeline-config.h"

#include <boost/optional.hpp>

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

/// The type for an environment variable name.
typedef std::string envvar_name_t;
/// The type of an environment variable value.
typedef boost::optional<std::string> envvar_value_t;

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
VISTK_PIPELINE_EXPORT bool name_thread(thread_name_t const& name);

/**
 * \brief Retrieve the value of an environment variable.
 *
 * \param name The variable to retrieve from the environement.
 *
 * \returns The value of the environment variable, \c NULL if it was not set.
 */
VISTK_PIPELINE_EXPORT envvar_value_t get_envvar(envvar_name_t const& name);

}

#endif // VISTK_PIPELINE_UTILS_H
