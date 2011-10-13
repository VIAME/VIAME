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
 * \def CREATE_INST
 *
 * \brief The name of the function which creates an instance of the process.
 *
 * \param name The base name of the process.
 */
#define CREATE_INST(name) create_##name##_process

/**
 * \def DECLARE_CREATE_INSTANCE
 *
 * \brief Declares the function to create the process.
 *
 * \param name The base name of the process.
 */
#define DECLARE_CREATE_INSTANCE(name) \
  static process_t CREATE_INST(name)(vistk::config_t const& config)

/**
 * \def REGISTER_CLASS
 *
 * \brief Registers the class with the registry.
 *
 * \param name The base name of the process.
 * \param desc A description of the process.
 */
#define REGISTER_CLASS(name, desc) \
  registry->register_process(#name, vistk::process_registry::description_t(desc), CREATE_INST(name))

/**
 * \def DEFINE_CREATE_INSTANCE
 *
 * \brief Implements the function which creates an instance of the process.
 *
 * \param name The base name of the process.
 */
#define DEFINE_CREATE_INSTANCE(name)                     \
  vistk::process_t                                       \
  CREATE_INST(name)(vistk::config_t const& config)       \
  {                                                      \
    return boost::make_shared<CLASS_NAME(name)>(config); \
  }

#endif // VISTK_PROCESSES_HELPER_FUNCTIONS_FUNCTION_PROCESS_REGISTER_H
