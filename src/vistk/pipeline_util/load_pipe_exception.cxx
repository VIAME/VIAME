/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "load_pipe_exception.h"

#include <sstream>

/**
 * \file load_pipe_exception.cxx
 *
 * \brief Implementation of exceptions used when loading a pipe declaration.
 */

namespace vistk
{

load_pipe_exception
::load_pipe_exception() throw()
  : pipeline_exception()
{
}

load_pipe_exception
::~load_pipe_exception() throw()
{
}

file_no_exist_exception
::file_no_exist_exception(path_t const& fname) throw()
  : load_pipe_exception()
  , m_fname(fname)
{
  std::stringstream sstr;

  sstr << "The file does not exist: " << m_fname;

  m_what = sstr.str();
}

file_no_exist_exception
::~file_no_exist_exception() throw()
{
}

not_a_file_exception
::not_a_file_exception(path_t const& path) throw()
  : load_pipe_exception()
  , m_path(path)
{
  std::stringstream sstr;

  sstr << "The path is not a file: " << m_path;

  m_what = sstr.str();
}

not_a_file_exception
::~not_a_file_exception() throw()
{
}

file_open_exception
::file_open_exception(path_t const& fname) throw()
  : load_pipe_exception()
  , m_fname(fname)
{
  std::stringstream sstr;

  sstr << "Failure when opening a file: " << m_fname;

  m_what = sstr.str();
}

file_open_exception
::~file_open_exception() throw()
{
}

stream_failure_exception
::stream_failure_exception(std::string const& msg) throw()
  : load_pipe_exception()
  , m_msg(msg)
{
  std::stringstream sstr;

  sstr << "Failure when using a stream: " << m_msg;

  m_what = sstr.str();
}

stream_failure_exception
::~stream_failure_exception() throw()
{
}

size_t const failed_to_parse::max_size = 64;

failed_to_parse
::failed_to_parse(std::string const& reason, std::string const& where) throw()
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
::~failed_to_parse() throw()
{
}

}
