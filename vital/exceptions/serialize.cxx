// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation for serialization exceptions
 */

#include "serialize.h"

namespace kwiver {
namespace vital {

serialization_exception
::serialization_exception( std::string const& str )
{
  m_what = str;
}

serialization_exception
::~serialization_exception() noexcept
{ }

} } // end vital namespace
