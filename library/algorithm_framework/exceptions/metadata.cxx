// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Implementation for metadata exceptions

#include "metadata.h"

namespace kwiver {

namespace vital {

metadata_exception
::metadata_exception( std::string const& str )
{
  m_what = str;
}

metadata_exception
::~metadata_exception() noexcept
{}

metadata_buffer_overflow
::metadata_buffer_overflow( std::string const& str ) : metadata_exception( str )
{}

metadata_buffer_overflow
::~metadata_buffer_overflow() noexcept
{}

metadata_type_overflow
::metadata_type_overflow( std::string const& str ) : metadata_exception( str )
{}

metadata_type_overflow
::~metadata_type_overflow() noexcept
{}

} // namespace vital

} // namespace kwiver
