/*ckwg +29
 * Copyright 2015 by Kitware, Inc.
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

#include "SMQTK_Descriptor.h"

#include <vital/config/config_block.h>

#include <sprokit/tools/pipeline_builder.h>
#include <sprokit/pipeline/modules.h>
#include <sprokit/pipeline/scheduler.h>
#include <sprokit/pipeline/scheduler_registry.h>
#include <sprokit/pipeline/pipeline.h>

#include "supply_image.h"
#include "accept_descriptor.h"
#include "io_mgr.h"

#include <iostream>
#include <sstream>

#include <cstdlib>

namespace kwiver {

static kwiver::vital::config_block_key_t const scheduler_block = kwiver::vital::config_block_key_t("_scheduler");


// ==================================================================
class SMQTK_Descriptor::priv
{
public:
  priv() {}


  ~priv() {}

};


// ==================================================================
SMQTK_Descriptor::
SMQTK_Descriptor()
  : d( new SMQTK_Descriptor::priv() )
{
}


SMQTK_Descriptor::
~SMQTK_Descriptor()
{
}


// ------------------------------------------------------------------
std::vector< double >
SMQTK_Descriptor::
ExtractSMQTK(  cv::Mat cv_img, std::string const& config_file )
{
  // 1) register input and processes with sprokit
  static sprokit::process_registry::module_t const module_name =
    sprokit::process_registry::module_t( "smqtk_processes" );

  sprokit::process_registry_t const registry( sprokit::process_registry::self() );

  if ( registry->is_module_loaded( module_name ) )
  {
    return std::vector< double >();
  }

  // 2) Make image available for input.
  kwiver::io_mgr::Instance()->SetImage( cv_img );

  // 3) ...

  // 4) locate python process and get it loaded

  sprokit::load_known_modules(); //+ maybe not needed

  // 5) create pipeline description
  std::stringstream pipeline_desc;
  pipeline_desc << "process input_endcap\n"
                << "  :: supply_image\n"
                << "process descriptor\n"
                << "  :: ApplyDescriptor\n"
                << "process output_endcap\n"
                << "  :: accept_descriptor\n"

                << "connect from input_endcap.image\n"
                << "          to descriptor.image\n"

                << "connect from descriptor.vector\n"
                << "          to output_endcap.d_vector\n"
    ;

  // 7) create a pipeline
  sprokit::pipeline_builder builder;
  builder.load_pipeline( pipeline_desc );

  // build pipeline
  sprokit::pipeline_t const pipe = builder.pipeline();
  if (!pipe)
  {
    std::cerr << "Error: Unable to bake pipeline" << std::endl;
    return std::vector< double >();
  }

  // For the next version, use the pipeline interface to locate
  // endcaps and interact with them.
  // process_t proc = pipeline->process_by_name( "" );
  // cast proc to real derived process type
  // interact with process.

  // perform setup operation on pipeline and get it ready to run
  // This throws many exceptions
  try
  {
    pipe->setup_pipeline();
  }
  catch( sprokit::pipeline_exception const& e)
  {
    std::cerr << "Error setting up pipeline: " << e.what() << std::endl;
    return std::vector< double >();
  }

  sprokit::scheduler_registry::type_t const scheduler_type = "pythread_per_process";

    // Get config from pipeline and force scheduler type
  kwiver::vital::config_block_sptr conf = builder.config();
  conf->print( std::cout );

  // Add config stream to descriptor process
  conf->set_value( "input_endcap:config_file", config_file );


  kwiver::vital::config_block_sptr const scheduler_config = conf->subblock(scheduler_block +
                                              kwiver::vital::config_block::block_sep + scheduler_type);

  sprokit::scheduler_registry_t reg = sprokit::scheduler_registry::self();

  sprokit::scheduler_t scheduler = reg->create_scheduler(scheduler_type, pipe, scheduler_config);

  if (!scheduler)
  {
    std::cerr << "Error: Unable to create scheduler" << std::endl;
    return std::vector< double >();
  }

  // Start pipeline and wait for it to finish
  scheduler->start();
  scheduler->wait();

  // Extract pipeline results
  kwiver::vital::double_vector_sptr d_ptr = kwiver::io_mgr::Instance()->GetDescriptor();
  return *d_ptr.get(); // return by value
}

} // end namespace
