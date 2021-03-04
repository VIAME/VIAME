// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "datum.h"

#include <sstream>

/**
 * \file datum.cxx
 *
 * \brief Implementation of a piece of \link sprokit::datum data\endlink in the pipeline.
 */

namespace sprokit {

// ------------------------------------------------------------------
datum_t
datum::new_datum(kwiver::vital::any const& dat)
{
  return datum_t(new datum(dat));
}

// ------------------------------------------------------------------
datum_t
datum
::empty_datum()
{
  return datum_t(new datum(empty));
}

// ------------------------------------------------------------------
datum_t
datum
::flush_datum()
{
  return datum_t(new datum(flush));
}

// ------------------------------------------------------------------
datum_t
datum
::complete_datum()
{
  return datum_t(new datum(complete));
}

// ------------------------------------------------------------------
datum_t
datum
::error_datum(error_t const& error)
{
  return datum_t(new datum(error));
}

// ------------------------------------------------------------------
datum::type_t
datum
::type() const
{
  return m_type;
}

// ------------------------------------------------------------------
datum::error_t
datum
::get_error() const
{
  return m_error;
}

static bool any_equal(kwiver::vital::any const& a, kwiver::vital::any const& b);

// ------------------------------------------------------------------
bool
datum
::operator == (datum const& dat) const
{
  if (this == &dat)
  {
    return true;
  }

  if (m_type != dat.m_type)
  {
    return false;
  }

  bool ret = false;

  switch (m_type)
  {
    case data:
      ret = any_equal(m_datum, dat.m_datum);
      break;

    case empty:
    case flush:
    case complete:
      ret = true;
      break;

    case error:
      ret = (m_error == dat.m_error);
      break;

    case invalid:
    default:
      ret = false;
      break;
  }

  return ret;
}

// ------------------------------------------------------------------
datum
::datum(type_t ty)
  : m_type(ty)
  , m_error()
  , m_datum()
{
}

// ------------------------------------------------------------------
datum
::datum(error_t const& err)
  : m_type(error)
  , m_error(err)
  , m_datum()
{
}

// ------------------------------------------------------------------
datum
::datum(kwiver::vital::any const& dat)
  : m_type(data)
  , m_error()
  , m_datum(dat)
{
}

// ------------------------------------------------------------------
datum_exception
::datum_exception() noexcept
  : pipeline_exception()
{
}

datum_exception
::~datum_exception() noexcept
{
}

static char const* string_for_type(datum::type_t type);

// ------------------------------------------------------------------
bad_datum_cast_exception
::bad_datum_cast_exception(std::string const& requested_typeid,
                           std::string const& typeid_,
                           datum::type_t const& type,
                           datum::error_t const& error,
                           char const* reason) noexcept
  : datum_exception()
  , m_requested_typeid(requested_typeid)
  , m_typeid(typeid_)
  , m_type(type)
  , m_error(error)
  , m_reason(reason)
{
  std::ostringstream sstr;

  if (m_type == datum::error)
  {
    sstr << "Failed to cast datum of type "
            "\'" << string_for_type(m_type) << "\' (" << m_error << ") into "
         << m_typeid << ": "
         << m_reason << ".";
  }
  else if (m_type == datum::data)
  {
    sstr << "Failed to cast datum of type "
            "\'" << string_for_type(m_type) << "\' (" << m_typeid << ") into "
         << m_requested_typeid << ": "
         << m_reason << ".";
  }
  else
  {
    sstr << "Failed to cast datum of type "
            "\'" << string_for_type(m_type) << "\' into "
         << m_requested_typeid << ": "
         << m_reason << ".";
  }

  m_what = sstr.str();
}

// ------------------------------------------------------------------
bad_datum_cast_exception
::~bad_datum_cast_exception() noexcept
{
}

// ------------------------------------------------------------------
bool
any_equal(kwiver::vital::any const& a, kwiver::vital::any const& b)
{
  if (a.empty() && b.empty())
  {
    return true;
  }

  if (a.type() != b.type())
  {
    return false;
  }

  // Be safe.
  return false;
}

// ------------------------------------------------------------------
char const*
string_for_type(datum::type_t type)
{
  switch (type)
  {
#define STRING_FOR_TYPE(type) \
  case datum::type:           \
    return #type

    STRING_FOR_TYPE(data);
    STRING_FOR_TYPE(empty);
    STRING_FOR_TYPE(error);
    STRING_FOR_TYPE(invalid);
    STRING_FOR_TYPE(flush);
    STRING_FOR_TYPE(complete);

#undef STRING_FOR_TYPE

    default:
      break;
  }

  return "unknown";
}

}
