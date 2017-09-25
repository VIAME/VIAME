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

#include <arrows/ocv/image_container.h>

#include <sprokit/pipeline/modules.h>
#include <sprokit/pipeline/pipeline.h>
#include <sprokit/pipeline/process_registry.h>
#include <sprokit/processes/kwiver_type_traits.h>
#include <sprokit/processes/adapters/embedded_pipeline.h>
#include <sprokit/tools/literal_pipeline.h>

namespace kwiver {

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
  // register processes with sprokit
  static sprokit::process_registry::module_t const module_name =
    sprokit::process_registry::module_t( "smqtk_processes" );


  sprokit::load_known_modules();

  // create pipeline description
  std::stringstream pipeline_desc;
  pipeline_desc << SPROKIT_PROCESS( "input_adapter",  "ia" )
                << SPROKIT_PROCESS( "output_adapter", "oa" )

                << SPROKIT_PROCESS( "process descriptor", "ApplyDescriptor" )

                << SPROKIT_CONFIG( "config_file",  config_file )

                << SPROKIT_CONNECT( "ia", "image",                "descriptor", "image" )
                << SPROKIT_CONNECT( "descriptor", "vector",       "oa", "d_vector" )
    ;

  // create a embedded pipeline
  kwiver::embedded_pipeline ep;
  ep.build_pipeline( pipeline_desc );

  // Start pipeline and wait for it to finish
  ep.start();

  // Create dataset for input
  auto ds = kwiver::adapter::adapter_data_set::create();

  // Put OCV image in vital container
  kwiver::vital::image_container_sptr img( new kwiver::arrows::ocv::image_container( cv_img ) );

  ds->add_value( "image", img );
  ep.send( ds );
  ep.send_end_of_input(); // indicate end of input

  // Get results from pipeline
  auto rds = ep.receive();
  auto ix = rds->find( "d_vector" );
  auto d_ptr = ix->second->get_datum<kwiver::vital::double_vector_sptr>();

  ep.wait();

  return *d_ptr.get(); // return by value
}

} // end namespace
