// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation for plugin exceptions
 */

#include "plugin.h"

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
plugin_exception
::plugin_exception() noexcept
{
}

plugin_exception
::~plugin_exception() noexcept
{
}

// ------------------------------------------------------------------
plugin_factory_not_found
::plugin_factory_not_found( std::string const& msg) noexcept
{
  m_what = msg;
}

plugin_factory_not_found
::~plugin_factory_not_found() noexcept
{
}

// ------------------------------------------------------------------
plugin_factory_type_creation_error
::plugin_factory_type_creation_error( std::string const& msg) noexcept
{
  m_what = msg;
}

plugin_factory_type_creation_error
::~plugin_factory_type_creation_error() noexcept
{
}

// ------------------------------------------------------------------
plugin_already_exists
::plugin_already_exists( std::string const& msg) noexcept
{
  m_what = msg;
}

plugin_already_exists
::~plugin_already_exists() noexcept
{
}

} } // end namespace
