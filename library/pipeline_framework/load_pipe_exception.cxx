// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
