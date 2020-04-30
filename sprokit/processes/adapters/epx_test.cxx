/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "epx_test.h"

#include <vital/plugin_loader/plugin_loader.h>
#include <vital/plugin_loader/plugin_manager.h>
#include <vital/config/config_block_formatter.h>

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
end_of_output( context& ctxt )
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
