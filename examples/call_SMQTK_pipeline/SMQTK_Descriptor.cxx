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

#include <iostream>

#include <cstdlib>


// ==================================================================
class SMQTK_Descriptor::priv
{
public:
  priv()
  {
    m_pipeline_config = "
process supply_image
:: image_source

process SMQTK_desc
:: ApplyDescriptor

process accept_vector
:: get_vector
";
  }

  ~priv() {}

  std::string m_pipeline_config;
};


// ==================================================================
SMQTK_Descriptor::
SMQTK_Descriptor()
  : priv( new SMQTK_Descriptor::priv() )
{
}


SMQTK_Descriptor::
~SMQTK_Descriptor()
{
}


// ------------------------------------------------------------------
std::vector< double >
SMQTK_Descriptor::
ExtractSMQTK(  cv::Mat cv_img, std::istream const& config )
{

  // 1) instantiate input process with input image

  // 2) instantiate output process

  // 3) register input and processes with sprokit
  static sprokit::process_registry::module_t const module_name =
    sprokit::process_registry::module_t( "smqtk_processes" );

  sprokit::process_registry_t const registry( sprokit::process_registry::self() );

  if ( registry->is_module_loaded( module_name ) )
  {
    return;
  }

  registry->register_process( "supply_image", "Supplies a single image",
    sprokit::create_process< supply_image > );

  registry->register_process( "accept_vector", "Reads a single vector",
    sprokit::create_process< accept_vector > );


  // 4) locate python process and get it loaded

  sprokit::load_known_modules(); //+ maybe not needed

  //+ VM is a problem
  sprokit::pipeline_builder const builder(vm, desc);

  sprokit::pipeline_t const pipe = builder.pipeline();
  kwiver::vital::config_block_sptr const conf = builder.config();

  if (!pipe)
  {
    std::cerr << "Error: Unable to bake pipeline" << std::endl;

    return EXIT_FAILURE;
  }

  pipe->setup_pipeline();

  sprokit::scheduler_registry::type_t const scheduler_type = "pythread_per_process";
  kwiver::vital::config_block_sptr const scheduler_config = conf->subblock(scheduler_block +
                                              kwiver::vital::config_block::block_sep + scheduler_type);

  sprokit::scheduler_registry_t reg = sprokit::scheduler_registry::self();

  sprokit::scheduler_t scheduler = reg->create_scheduler(scheduler_type, pipe, scheduler_config);

  if (!scheduler)
  {
    std::cerr << "Error: Unable to create scheduler" << std::endl;

    return EXIT_FAILURE;
  }

  scheduler->start();
  scheduler->wait();

}
