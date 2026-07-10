/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "vertex_ai_trainer.h"
#include "vertex_ai_client.h"

#include <sprokit/processes/kwiver_type_traits.h>

#include <sstream>

namespace viame {
namespace vertex_ai {

// Config traits
create_config_trait( project_id, std::string, "",
  "Google Cloud project ID" );
create_config_trait( location, std::string, "us-central1",
  "GCP region" );
create_config_trait( credentials_file, std::string, "",
  "Path to service account JSON key" );
create_config_trait( container_image, std::string, "",
  "Container image URI in Artifact Registry" );
create_config_trait( container_args, std::string, "",
  "Comma-separated arguments to pass to the container entrypoint" );
create_config_trait( machine_type, std::string, "n1-standard-4",
  "Compute Engine machine type" );
create_config_trait( accelerator_type, std::string, "",
  "GPU type (e.g. NVIDIA_TESLA_T4, NVIDIA_TESLA_V100)" );
create_config_trait( accelerator_count, int, "0",
  "Number of GPUs" );
create_config_trait( output_gcs_dir, std::string, "",
  "GCS URI for training output (gs://bucket/path)" );
create_config_trait( wait_for_completion, bool, "true",
  "Block until the training job completes" );
create_config_trait( poll_interval_sec, int, "30",
  "Seconds between job status polls" );
create_config_trait( job_display_name, std::string, "viame-training-job",
  "Display name for the training job" );

class vertex_ai_trainer::priv
{
public:
  priv() {}
  ~priv() {}

  std::string project_id;
  std::string location;
  std::string credentials_file;
  std::string container_image;
  std::string container_args_str;
  std::string machine_type;
  std::string accelerator_type;
  int accelerator_count = 0;
  std::string output_gcs_dir;
  bool wait_for_completion = true;
  int poll_interval_sec = 30;
  std::string job_display_name;

  vertex_ai_client client;
  bool job_submitted = false;
};

vertex_ai_trainer::vertex_ai_trainer(
  kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new priv )
{
  make_ports();
  make_config();
}

vertex_ai_trainer::~vertex_ai_trainer()
{}

void vertex_ai_trainer::_configure()
{
  d->project_id = config_value_using_trait( project_id );
  d->location = config_value_using_trait( location );
  d->credentials_file = config_value_using_trait( credentials_file );
  d->container_image = config_value_using_trait( container_image );
  d->container_args_str = config_value_using_trait( container_args );
  d->machine_type = config_value_using_trait( machine_type );
  d->accelerator_type = config_value_using_trait( accelerator_type );
  d->accelerator_count = config_value_using_trait( accelerator_count );
  d->output_gcs_dir = config_value_using_trait( output_gcs_dir );
  d->wait_for_completion = config_value_using_trait( wait_for_completion );
  d->poll_interval_sec = config_value_using_trait( poll_interval_sec );
  d->job_display_name = config_value_using_trait( job_display_name );

  d->client.set_project( d->project_id );
  d->client.set_location( d->location );

  if( !d->credentials_file.empty() )
  {
    d->client.set_credentials_file( d->credentials_file );
  }

  if( !d->client.authenticate() )
  {
    throw sprokit::invalid_configuration_exception(
      name(), "Failed to authenticate with GCP" );
  }
}

void vertex_ai_trainer::_step()
{
  if( d->job_submitted )
  {
    mark_process_as_complete();
    return;
  }

  // Parse container args
  std::vector< std::string > args;

  if( !d->container_args_str.empty() )
  {
    std::istringstream ss( d->container_args_str );
    std::string token;

    while( std::getline( ss, token, ',' ) )
    {
      if( !token.empty() )
      {
        args.push_back( token );
      }
    }
  }

  // Optionally read a GCS input path from a port
  if( has_input_port_edge( "input_gcs_dir" ) )
  {
    std::string input_dir =
      grab_from_port_as< std::string >( "input_gcs_dir" );
    args.push_back( "--input-dir=" + input_dir );
  }

  auto job = d->client.submit_training_job(
    d->job_display_name,
    d->container_image,
    args,
    d->machine_type,
    d->accelerator_type,
    d->accelerator_count,
    d->output_gcs_dir );

  if( job.state == "SUBMIT_FAILED" )
  {
    LOG_ERROR( logger(),
      "Training job failed to submit: " << job.error_message );
    push_to_port_as< std::string >( "job_state", "FAILED" );
    d->job_submitted = true;
    return;
  }

  LOG_INFO( logger(), "Training job submitted: " << job.job_name );

  if( d->wait_for_completion )
  {
    job = d->client.wait_for_job(
      job.job_name, d->poll_interval_sec );
    LOG_INFO( logger(),
      "Training job finished: " << job.state );
  }

  push_to_port_as< std::string >( "job_name", job.job_name );
  push_to_port_as< std::string >( "job_state", job.state );

  d->job_submitted = true;
}

void vertex_ai_trainer::make_ports()
{
  sprokit::process::port_flags_t optional_;

  declare_input_port(
    "input_gcs_dir", "kwiver:string", optional_,
    "Optional GCS URI for training input data" );

  declare_output_port(
    "job_name", "kwiver:string", optional_,
    "Vertex AI job resource name" );
  declare_output_port(
    "job_state", "kwiver:string", optional_,
    "Final job state (JOB_STATE_SUCCEEDED, etc.)" );
}

void vertex_ai_trainer::make_config()
{
  declare_config_using_trait( project_id );
  declare_config_using_trait( location );
  declare_config_using_trait( credentials_file );
  declare_config_using_trait( container_image );
  declare_config_using_trait( container_args );
  declare_config_using_trait( machine_type );
  declare_config_using_trait( accelerator_type );
  declare_config_using_trait( accelerator_count );
  declare_config_using_trait( output_gcs_dir );
  declare_config_using_trait( wait_for_completion );
  declare_config_using_trait( poll_interval_sec );
  declare_config_using_trait( job_display_name );
}

} // namespace vertex_ai
} // namespace viame
