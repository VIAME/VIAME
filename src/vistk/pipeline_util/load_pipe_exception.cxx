/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
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

file_no_exist_exception
::file_no_exist_exception(boost::filesystem::path const& fname) throw()
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

char const*
file_no_exist_exception
::what() const throw()
{
  return m_what.c_str();
}

file_open_exception
::file_open_exception(boost::filesystem::path const& fname) throw()
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

char const*
file_open_exception
::what() const throw()
{
  return m_what.c_str();
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

char const*
stream_failure_exception
::what() const throw()
{
  return m_what.c_str();
}

failed_to_parse
::failed_to_parse(std::string const& reason, std::string const& where) throw()
  : load_pipe_exception()
  , m_reason(reason)
  , m_where(where)
{
  std::stringstream sstr;

  sstr << "Expected: \'" << m_reason << "\' "
       << "when \'" << m_where << "\' was given";

  m_what = sstr.str();
}

failed_to_parse
::~failed_to_parse() throw()
{
}

char const*
failed_to_parse
::what() const throw()
{
  return m_what.c_str();
}

} // end namespace vistk
