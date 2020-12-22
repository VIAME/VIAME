// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation for image exceptions
 */

#include "image.h"

#include <sstream>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------------------
image_exception
::image_exception( std::string const& message ) noexcept
{
  m_what = message;
}

// ----------------------------------------------------------------------------
image_exception
::image_exception( std::nullptr_t ) noexcept
{
}

// ----------------------------------------------------------------------------
image_exception
::~image_exception() noexcept
{
}

// ----------------------------------------------------------------------------
image_type_mismatch_exception
::image_type_mismatch_exception( std::string const& message ) noexcept
  : image_exception{ message }
{
}

// ----------------------------------------------------------------------------
image_type_mismatch_exception
::~image_type_mismatch_exception() noexcept
{
}

// ----------------------------------------------------------------------------
image_size_mismatch_exception
::image_size_mismatch_exception( std::string const& message,
                                 size_t correct_w, size_t correct_h,
                                 size_t given_w, size_t given_h ) noexcept
  : image_exception{ nullptr },
    m_message{ message },
    m_correct_w{ correct_w },
    m_correct_h{ correct_h },
    m_given_w{ given_w },
    m_given_h{ given_h }
{
  std::ostringstream ss;
  ss << message
     << " (given: [" << given_w << ", " << given_h << "],"
     << " should be: [" << correct_w << ", " << correct_h << "])";
  m_what = ss.str();
}

// ----------------------------------------------------------------------------
image_size_mismatch_exception
::~image_size_mismatch_exception() noexcept
{
}

} // end namespace vital
} // end namespace kwiver
