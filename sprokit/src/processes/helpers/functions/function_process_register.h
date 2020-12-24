// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file function_process_register.h
 *
 * \brief Macros for registering a process which wraps a function.
 */

#ifndef SPROKIT_PROCESSES_HELPER_FUNCTIONS_FUNCTION_PROCESS_REGISTER_H
#define SPROKIT_PROCESSES_HELPER_FUNCTIONS_FUNCTION_PROCESS_REGISTER_H

#include "function_process.h"
#include <sprokit/pipeline/process_factory.h>

/**
 * \def REGISTER_FUNCTION
 *
 * \brief Registers the class with the registry.
 *
 * \param name The base name of the process.
 * \param desc A description of the process.
 */
#define REGISTER_FUNCTION(name, desc)                                   \
  kwiver::vital::plugin_factory_handle_t fact = vpm.ADD_PROCESS( name ); \
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, #name ); \
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, desc );

#endif // SPROKIT_PROCESSES_HELPER_FUNCTIONS_FUNCTION_PROCESS_REGISTER_H
