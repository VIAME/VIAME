// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "kwiver_logger_factory.h"

namespace kwiver {
namespace vital {
namespace logger_ns {

// ----------------------------------------------------------------
/**
 *
 *
 */
kwiver_logger_factory
::kwiver_logger_factory( std::string const& name )
  :m_name(name)
{ }

kwiver_logger_factory
::~kwiver_logger_factory()
{ }

// ------------------------------------------------------------------
// Get location strings
std::string const & kwiver_logger_factory
::get_factory_name() const
{
  return m_name;
}

} } } // end namespace
