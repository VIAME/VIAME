/*ckwg +29
 * Copyright 2011-2016 by Kitware, Inc.
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
