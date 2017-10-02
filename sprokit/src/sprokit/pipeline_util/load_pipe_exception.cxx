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
 * \file load_pipe_exception.cxx
 *
 * \brief Implementation of exceptions used when loading a pipe declaration.
 */

#include "load_pipe_exception.h"

#include <sstream>

namespace sprokit
{

load_pipe_exception
::load_pipe_exception() noexcept
  : pipeline_exception()
{
}

load_pipe_exception
::~load_pipe_exception() noexcept
{
}


// ------------------------------------------------------------------
file_no_exist_exception
::file_no_exist_exception( kwiver::vital::path_t const& fname) noexcept
  : load_pipe_exception()
  , m_fname(fname)
{
  std::stringstream sstr;
  sstr << "The file does not exist: " << m_fname;

  m_what = sstr.str();
}

file_no_exist_exception
::~file_no_exist_exception() noexcept
{
}


// ------------------------------------------------------------------
parsing_exception::
parsing_exception( const std::string& msg ) noexcept
: load_pipe_exception()
{
  m_what = msg;
}


parsing_exception::
~parsing_exception() noexcept
{ }

}
