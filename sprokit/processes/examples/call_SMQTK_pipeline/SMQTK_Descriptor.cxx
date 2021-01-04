// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
