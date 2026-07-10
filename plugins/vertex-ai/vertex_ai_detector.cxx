/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "vertex_ai_detector.h"
#include "vertex_ai_client.h"

#include <sprokit/processes/kwiver_type_traits.h>

#include <vital/types/image_container.h>
#include <vital/types/detected_object_set.h>
#include <vital/types/descriptor_set.h>

#include <sstream>

namespace viame {
namespace vertex_ai {

// Config traits
create_config_trait( project_id, std::string, "",
  "Google Cloud project ID" );
create_config_trait( location, std::string, "us-central1",
  "GCP region for the Vertex AI endpoint" );
create_config_trait( endpoint_id, std::string, "",
  "Deployed Vertex AI endpoint ID" );
create_config_trait( credentials_file, std::string, "",
  "Path to GCP service account JSON key (empty = use ADC)" );
create_config_trait( timeout_seconds, int, "120",
  "HTTP request timeout in seconds" );
create_config_trait( input_format, std::string, "descriptor",
  "How to serialize input: 'descriptor' (send descriptor vectors), "
  "'image_base64' (send base64-encoded image), "
  "'custom' (use raw_json_input port)" );

class vertex_ai_detector::priv
{
public:
  priv() {}
  ~priv() {}

  std::string project_id;
  std::string location;
  std::string endpoint_id;
  std::string credentials_file;
  std::string input_format;
  int timeout_seconds = 120;

  vertex_ai_client client;
  bool authenticated = false;
};

vertex_ai_detector::vertex_ai_detector(
  kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new priv )
{
  make_ports();
  make_config();
}

vertex_ai_detector::~vertex_ai_detector()
{}

void vertex_ai_detector::_configure()
{
  d->project_id = config_value_using_trait( project_id );
  d->location = config_value_using_trait( location );
  d->endpoint_id = config_value_using_trait( endpoint_id );
  d->credentials_file = config_value_using_trait( credentials_file );
  d->input_format = config_value_using_trait( input_format );
  d->timeout_seconds = config_value_using_trait( timeout_seconds );

  d->client.set_project( d->project_id );
  d->client.set_location( d->location );
  d->client.set_timeout_seconds( d->timeout_seconds );

  if( !d->credentials_file.empty() )
  {
    d->client.set_credentials_file( d->credentials_file );
  }

  d->authenticated = d->client.authenticate();

  if( !d->authenticated )
  {
    throw sprokit::invalid_configuration_exception(
      name(), "Failed to authenticate with GCP" );
  }
}

void vertex_ai_detector::_step()
{
  std::string instances_json;

  if( d->input_format == "descriptor" )
  {
    auto desc_set = grab_from_port_using_trait( descriptor_set );

    std::ostringstream oss;
    oss << "[";

    if( desc_set )
    {
      auto descs = desc_set->descriptors();

      for( size_t i = 0; i < descs.size(); ++i )
      {
        if( i > 0 ) oss << ",";
        oss << "[";

        auto d_ptr = descs[i]->as_double();

        for( size_t j = 0; j < descs[i]->size(); ++j )
        {
          if( j > 0 ) oss << ",";
          oss << d_ptr[j];
        }

        oss << "]";
      }
    }

    oss << "]";
    instances_json = oss.str();
  }
  else if( d->input_format == "image_base64" )
  {
    auto image = grab_from_port_using_trait( image );
    // TODO: encode image to PNG/JPEG bytes then base64
    instances_json = "[{\"image\":{\"bytesBase64Encoded\":\"\"}}]";
  }
  else if( d->input_format == "custom" )
  {
    instances_json =
      grab_from_port_as< std::string >( "raw_json_input" );
  }

  auto results = d->client.predict(
    d->endpoint_id, instances_json );

  if( !results.empty() )
  {
    push_to_port_as< std::string >(
      "raw_json_output", results[0].raw_json );

    // TODO: parse model-specific response into detected_object_set
    auto det_set =
      std::make_shared< kwiver::vital::detected_object_set >();
    push_to_port_using_trait( detected_object_set, det_set );
  }
  else
  {
    push_to_port_as< std::string >( "raw_json_output", "" );
    push_to_port_using_trait( detected_object_set,
      std::make_shared< kwiver::vital::detected_object_set >() );
  }
}

void vertex_ai_detector::make_ports()
{
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional_;

  required.insert( flag_required );

  // Inputs (one required depending on input_format)
  declare_input_port_using_trait( image, optional_ );
  declare_input_port_using_trait( descriptor_set, optional_ );
  declare_input_port(
    "raw_json_input", "kwiver:string", optional_,
    "Raw JSON instances for custom input format" );

  // Outputs
  declare_output_port_using_trait( detected_object_set, optional_ );
  declare_output_port(
    "raw_json_output", "kwiver:string", optional_,
    "Raw JSON response from Vertex AI" );
}

void vertex_ai_detector::make_config()
{
  declare_config_using_trait( project_id );
  declare_config_using_trait( location );
  declare_config_using_trait( endpoint_id );
  declare_config_using_trait( credentials_file );
  declare_config_using_trait( timeout_seconds );
  declare_config_using_trait( input_format );
}

} // namespace vertex_ai
} // namespace viame
