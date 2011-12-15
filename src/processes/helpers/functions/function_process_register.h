/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_HELPER_FUNCTIONS_FUNCTION_PROCESS_REGISTER_H
#define VISTK_PROCESSES_HELPER_FUNCTIONS_FUNCTION_PROCESS_REGISTER_H

#include "function_process.h"

/**
 * \file function_process_register.h
 *
 * \brief Macros for registering a process which wraps a function.
 */

/**
 * \def REGISTER_FUNCTION
 *
 * \brief Registers the class with the registry.
 *
 * \param name The base name of the process.
 * \param desc A description of the process.
 */
#define REGISTER_FUNCTION(name, desc) \
  registry->register_process(#name, vistk::process_registry::description_t(desc), CREATE_PROCESS(CLASS_NAME(name)))

#endif // VISTK_PROCESSES_HELPER_FUNCTIONS_FUNCTION_PROCESS_REGISTER_H
