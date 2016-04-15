/*ckwg +29
 * Copyright 2011-2013 by Kitware, Inc.
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

#include "load_pipe_exception.h"

#include <sstream>

/**
 * \file load_pipe_exception.cxx
 *
 * \brief Implementation of exceptions used when loading a pipe declaration.
 */

namespace sprokit
{

load_pipe_exception
::load_pipe_exception() SPROKIT_NOTHROW
  : pipeline_exception()
{
}

load_pipe_exception
::~load_pipe_exception() SPROKIT_NOTHROW
{
}

file_no_exist_exception
::file_no_exist_exception(path_t const& fname) SPROKIT_NOTHROW
  : load_pipe_exception()
  , m_fname(fname)
{
  std::stringstream sstr;

  sstr << "The file does not exist: " << m_fname;

  m_what = sstr.str();
}

file_no_exist_exception
::~file_no_exist_exception() SPROKIT_NOTHROW
{
}

not_a_file_exception
::not_a_file_exception(path_t const& path) SPROKIT_NOTHROW
  : load_pipe_exception()
  , m_path(path)
{
  std::stringstream sstr;

  sstr << "The path is not a file: " << m_path;

  m_what = sstr.str();
}

not_a_file_exception
::~not_a_file_exception() SPROKIT_NOTHROW
{
}

file_open_exception
::file_open_exception(path_t const& fname) SPROKIT_NOTHROW
  : load_pipe_exception()
  , m_fname(fname)
{
  std::stringstream sstr;

  sstr << "Failure when opening a file: " << m_fname;

  m_what = sstr.str();
}

file_open_exception
::~file_open_exception() SPROKIT_NOTHROW
{
}

stream_failure_exception
::stream_failure_exception(std::string const& msg) SPROKIT_NOTHROW
  : load_pipe_exception()
  , m_msg(msg)
{
  std::stringstream sstr;

  sstr << "Failure when using a stream: " << m_msg;

  m_what = sstr.str();
}

stream_failure_exception
::~stream_failure_exception() SPROKIT_NOTHROW
{
}

size_t const failed_to_parse::max_size = 64;

failed_to_parse
::failed_to_parse(std::string const& reason, std::string const& where) SPROKIT_NOTHROW
  : load_pipe_exception()
  , m_reason(reason)
  , m_where_full(where)
  , m_where_brief(where.substr(0, max_size))
{
  std::stringstream sstr;

  sstr << "Expected: \'" << m_reason << "\' "
          "when \'" << m_where_brief << "\' was given";

  m_what = sstr.str();
}

failed_to_parse
::~failed_to_parse() SPROKIT_NOTHROW
{
}

}
