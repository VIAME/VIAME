/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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
 * \brief Interface to pipe parser exception
 */

#ifndef SPROKIT_PIPELINE_UTIL_PIPE_PARSER_EXCEPTION_H
#define SPROKIT_PIPELINE_UTIL_PIPE_PARSER_EXCEPTION_H

#include "pipeline_util-config.h"

#include <vital/vital_config.h>

#include <string>
#include <stdexcept>

namespace sprokit {

class SPROKIT_PIPELINE_UTIL_EXPORT parsing_exception
  : public std::exception
{
  public:
    /**
     * \brief Constructor.
     */
  parsing_exception( const std::string& msg) VITAL_NOTHROW;

    /**
     * \brief Destructor.
     */
    virtual ~parsing_exception() VITAL_NOTHROW;

    /**
     * \brief A description of the exception.
     *
     * \returns A string describing what went wrong.
     */
    const char* what() const VITAL_NOTHROW;

protected:
    /// The text of the exception.
    std::string m_what;
};

} // end namespace

#endif /* SPROKIT_PIPELINE_UTIL_PIPE_PARSER_EXCEPTION_H */
