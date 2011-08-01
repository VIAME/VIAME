/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "datum.h"

#include <sstream>

/**
 * \file datum.cxx
 *
 * \brief Implementation of a piece of \link vistk::datum data\endlink in the pipeline.
 */

namespace vistk
{

datum_t
datum
::empty_datum()
{
  return datum_t(new datum(false));
}

datum_t
datum
::complete_datum()
{
  return datum_t(new datum(true));
}

datum_t
datum
::error_datum(error_t const& error)
{
  return datum_t(new datum(error));
}

datum::datum_type_t
datum
::type() const
{
  return m_type;
}

datum::error_t
datum
::get_error() const
{
  return m_error;
}

datum
::datum(bool is_complete)
  : m_type(is_complete ? DATUM_EMPTY : DATUM_COMPLETE)
{
}

datum
::datum(error_t const& error)
  : m_type(DATUM_ERROR)
  , m_error(error)
{
}

datum
::datum(boost::any const& dat)
  : m_type(DATUM_DATA)
  , m_datum(dat)
{
}

bad_datum_cast_exception
::bad_datum_cast_exception(datum::datum_type_t const& type, char const* reason) throw()
  : datum_exception()
  , m_type(type)
  , m_reason(reason)
{
  std::ostringstream sstr;

  sstr << "Failed to cast key datum of type "
       << "\'" << m_type << "\': " << m_reason << ".";

  m_what = sstr.str();
}

bad_datum_cast_exception
::~bad_datum_cast_exception() throw()
{
}

char const*
bad_datum_cast_exception
::what() const throw()
{
  return m_what.c_str();
}

} // end namespace vistk
