// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "plugin_factory.h"

namespace kwiver {
namespace vital {

const std::string plugin_factory::INTERFACE_TYPE( "interface-type" );
const std::string plugin_factory::CONCRETE_TYPE( "concrete-type" );
const std::string plugin_factory::PLUGIN_FILE_NAME( "plugin-file-name" );
const std::string plugin_factory::PLUGIN_NAME( "plugin-name" );
const std::string plugin_factory::PLUGIN_DESCRIPTION( "plugin-descrip" );
const std::string plugin_factory::PLUGIN_VERSION( "plugin-version" );
const std::string plugin_factory::PLUGIN_MODULE_NAME( "plugin-version-name" );
const std::string plugin_factory::PLUGIN_FACTORY_TYPE( "plugin-factory-type" );
const std::string plugin_factory::PLUGIN_AUTHOR( "plugin-author" );
const std::string plugin_factory::PLUGIN_ORGANIZATION( "plugin-organization" );
const std::string plugin_factory::PLUGIN_LICENSE( "plugin-license" );
const std::string plugin_factory::PLUGIN_CATEGORY( "plugin-category" );
const std::string plugin_factory::PLUGIN_PROCESS_PROPERTIES( "plugin-process-properties" );

const std::string plugin_factory::APPLET_CATEGORY( "kwiver-applet" );
const std::string plugin_factory::PROCESS_CATEGORY( "process" );
const std::string plugin_factory::ALGORITHM_CATEGORY( "algorithm" );
const std::string plugin_factory::CLUSTER_CATEGORY( "cluster" );

// ------------------------------------------------------------------
plugin_factory::
plugin_factory( std::string const& itype )
{
  m_interface_type = itype; // Optimize and store locally
  add_attribute( INTERFACE_TYPE, itype );
}

plugin_factory::
~plugin_factory()
{ }

// ------------------------------------------------------------------
bool plugin_factory::
get_attribute( std::string const& attr, std::string& val ) const
{
  auto const it = m_attribute_map.find( attr );
  if ( it != m_attribute_map.end() )
  {
    val = it->second;
    return true;
  }

  return false;
}

// ------------------------------------------------------------------
plugin_factory&
plugin_factory::
add_attribute( std::string const& attr, std::string const& val )
{
  // Create if not there. Overwrite if already there.
  m_attribute_map[attr] = val;

  return *this;
}

} } // end namespace
