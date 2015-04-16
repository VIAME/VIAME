/*ckwg +29
 * Copyright 2014 by Kitware, Inc.
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
 * \file
 * \brief algorithm exceptions interfaces
 */

#ifndef KWIVER_CORE_EXCEPTIONS_ALGORITHM_H
#define KWIVER_CORE_EXCEPTIONS_ALGORITHM_H

#include "base.h"
#include <string>

namespace kwiver
{

/// Base class for all algorithm related exceptions
class KWIVER_CORE_EXPORT algorithm_exception
  : public kwiver_core_base_exception
{
  public:
    /// Constructor
    algorithm_exception(std::string type,
                        std::string impl,
                        std::string reason) KWIVER_NOTHROW;
    /// Deconstructor
    virtual ~algorithm_exception() KWIVER_NOTHROW;

    /// The name of the algorithm type
    std::string m_algo_type;
    /// The name of the algorithm implementation
    std::string m_algo_impl;
    /// String explanation of the reason for the exception
    std::string m_reason;
};

/// Exception for when an algorithm receives an invalid configuration
class KWIVER_CORE_EXPORT algorithm_configuration_exception
  : public algorithm_exception
{
  public:
    /// Constructor
    algorithm_configuration_exception(std::string type,
                                      std::string impl,
                                      std::string reason) KWIVER_NOTHROW;
    /// Destructor
    virtual ~algorithm_configuration_exception() KWIVER_NOTHROW;
};

/// Exception for when checking an invalid impl name against an algo def
class KWIVER_CORE_EXPORT invalid_name_exception
  : public algorithm_exception
{
  public:
    /// Constructor
    invalid_name_exception(std::string type,
                           std::string impl) KWIVER_NOTHROW;

    /// Destructor
    virtual ~invalid_name_exception() KWIVER_NOTHROW;
};

} // end namespace kwiver

#endif // KWIVER_CORE_EXCEPTIONS_ALGORITHM_H
