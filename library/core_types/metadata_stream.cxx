// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// Definition of the metadata stream classes.

#include <viame/core_types/metadata_stream.h>

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
metadata_stream
::metadata_stream()
{}

// ----------------------------------------------------------------------------
metadata_stream
::~metadata_stream()
{}

// ----------------------------------------------------------------------------
std::string
metadata_stream
::uri() const
{
  return {};
}

// ----------------------------------------------------------------------------
config_block_sptr
metadata_stream
::config() const
{
  return nullptr;
}

// ----------------------------------------------------------------------------
metadata_istream
::metadata_istream()
{}

// ----------------------------------------------------------------------------
metadata_istream
::~metadata_istream()
{}

// ----------------------------------------------------------------------------
metadata_ostream
::metadata_ostream()
{}

// ----------------------------------------------------------------------------
metadata_ostream
::~metadata_ostream()
{}

} // namespace vital

} // namespace kwiver
