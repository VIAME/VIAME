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

/**
 * \file load_pipe_exception.h
 *
 * \brief Header for exceptions used when loading a pipe declaration.
 */

#ifndef SPROKIT_PIPELINE_UTIL_LOAD_PIPE_EXCEPTION_H
#define SPROKIT_PIPELINE_UTIL_LOAD_PIPE_EXCEPTION_H

#include<sprokit/pipeline_util/sprokit_pipeline_util_export.h>

#include <vital/vital_types.h>

#include <sprokit/pipeline/types.h>

#include <string>
#include <cstddef>

namespace sprokit {

/**
 * \class load_pipe_exception load_pipe_exception.h <sprokit/pipeline_util/load_pipe_exception.h>
 *
 * \brief The base class for all exceptions thrown when loading a pipe declaration.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_UTIL_EXPORT load_pipe_exception
  : public pipeline_exception
{
  public:
    /**
     * \brief Constructor.
     */
    load_pipe_exception() noexcept;

    /**
     * \brief Destructor.
     */
    virtual ~load_pipe_exception() noexcept;
};

/**
 * \class file_no_exist_exception load_pipe_exception.h <sprokit/pipeline_util/load_pipe_exception.h>
 *
 * \brief The exception thrown when a file does not exist.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_UTIL_EXPORT file_no_exist_exception
  : public load_pipe_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param fname The path that does not exist.
     */
  file_no_exist_exception(kwiver::vital::path_t const& fname) noexcept;
    /**
     * \brief Destructor.
     */
    virtual ~file_no_exist_exception() noexcept;

    /// The path that does not exist.
    kwiver::vital::path_t const m_fname;
};


// ------------------------------------------------------------------
class SPROKIT_PIPELINE_UTIL_EXPORT parsing_exception
  : public load_pipe_exception
{
public:
  /**
   * \brief Constructor.
   */
  parsing_exception( const std::string& msg) noexcept;

  /**
   * \brief Destructor.
   */
  virtual ~parsing_exception() noexcept;

};

}

#endif // SPROKIT_PIPELINE_UTIL_LOAD_PIPE_EXCEPTION_H
