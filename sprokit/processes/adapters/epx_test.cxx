// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "epx_test.h"

#include <vital/plugin_loader/plugin_loader.h>
#include <vital/plugin_loader/plugin_manager.h>
#include <vital/config/config_block_formatter.h>
#include <vital/vital_config.h>

#include <sstream>
#include <iostream>

namespace kwiver {

epx_test::
epx_test()
{
}

// ----------------------------------------------------------------------------
void
epx_test::
pre_setup( context& ctxt )
{
  std::cout << "exp_test: Pre_Setup called\n";

  // print config
  std::stringstream str;

  vital::config_block_formatter fmt( ctxt.pipe_config() );
  fmt.print( str );

  std::cout <<  "exp_test: pipe config:\n" << str.str() <<std::endl;
}

// ----------------------------------------------------------------------------
void
epx_test::
end_of_output( VITAL_UNUSED context& ctxt )
{
  std::cout << "exp_test End_Of_Output called\n";
}

// ----------------------------------------------------------------------------
void epx_test::configure( kwiver::vital::config_block_sptr const conf )
{
  // print config
  std::stringstream str;
  vital::config_block_formatter fmt( conf);
  fmt.print( str );

  std::cout <<  "exp_test: configure called with config:\n" << str.str() <<std::endl;
}

// ----------------------------------------------------------------------------
kwiver::vital::config_block_sptr epx_test::get_configuration() const
{
  auto conf = kwiver::vital::config_block::empty_config();
  conf->set_value( "def-key1", "def_val" );
  conf->set_value( "one", "def_one" );

  return conf;
}

} // end namespace

// ----------------------------------------------------------------
/*! \brief Regsiter Extension
 *
 *
 */
extern "C"
KWIVER_EPX_TEST_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )

{
  kwiver::embedded_pipeline_extension_registrar reg( vpm, "kwiver_epx_test" );
  using namespace kwiver;

  if ( reg.is_module_loaded() )
  {
    return;
  }

  reg.register_EPX< epx_test >();

  reg.mark_module_as_loaded();
}
