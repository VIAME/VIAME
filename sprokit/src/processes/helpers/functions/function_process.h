/*ckwg +29
 * Copyright 2011-2017 by Kitware, Inc.
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

#ifndef SPROKIT_PROCESSES_HELPER_FUNCTIONS_FUNCTION_PROCESS_H
#define SPROKIT_PROCESSES_HELPER_FUNCTIONS_FUNCTION_PROCESS_H

/**
 * \file function_process.h
 *
 * \brief Macros for declaring processes which wrap functions.
 */

/**
 * \def CLASS_NAME
 *
 * \brief The class name of a function process given the base name.
 *
 * \param cls The name of the class.
 */
#define CLASS_NAME(cls) function_##cls##_process
/**
 * \def CLASS_DTOR
 *
 * \brief The destructor name of a function process given the base name.
 *
 * \param cls The name of the class.
 */
#define CLASS_DTOR(cls) ~function_##cls##_process

/**
 * \def DECLARE_FUNCTION_PROCESS
 *
 * \brief Declares a class which calls a function when stepped.
 *
 * \param name The base name of the class.
 */
#define DECLARE_FUNCTION_PROCESS(name)               \
class SPROKIT_NO_EXPORT CLASS_NAME(name)               \
  : public sprokit::process                            \
{                                                    \
  public:                                            \
    CLASS_NAME(name)(kwiver::vital::config_block_sptr const& config); \
    CLASS_DTOR(name)();                              \
  protected:                                         \
    void _configure();                               \
    void _step();                                    \
  private:                                           \
    class priv;                                      \
    std::unique_ptr<priv> d;                       \
}

#endif // SPROKIT_PROCESSES_HELPER_FUNCTIONS_FUNCTION_PROCESS_H
