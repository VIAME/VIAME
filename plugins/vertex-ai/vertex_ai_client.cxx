/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "vertex_ai_client.h"

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <httplib.h>

#include <sstream>
#include <thread>
#include <chrono>
#include <cstdio>

namespace viame {
namespace vertex_ai {

class vertex_ai_client::priv
{
public:
  std::string project_id;
  std::string location = "us-central1";
  std::string credentials_file;
  std::string access_token;
  int timeout_seconds = 120;
  kwiver::vital::logger_handle_t logger;

  priv()
    : logger( kwiver::vital::get_logger( "viame.vertex_ai.client" ) )
  {}

  std::string api_host() const
  {
    return location + "-aiplatform.googleapis.com";
  }

  httplib::Result post_json( const std::string& path,
                             const std::string& body )
  {
    httplib::SSLClient cli( api_host(), 443 );
    cli.set_read_timeout( timeout_seconds );
    httplib::Headers headers = {
      { "Authorization", "Bearer " + access_token },
      { "Content-Type", "application/json" }
    };
    return cli.Post( path.c_str(), headers, body, "application/json" );
  }

  httplib::Result get_json( const std::string& path )
  {
    httplib::SSLClient cli( api_host(), 443 );
    cli.set_read_timeout( timeout_seconds );
    httplib::Headers headers = {
      { "Authorization", "Bearer " + access_token }
    };
    return cli.Get( path.c_str(), headers );
  }
};

vertex_ai_client::vertex_ai_client()
  : d( new priv )
{}

vertex_ai_client::~vertex_ai_client()
{}

void vertex_ai_client::set_project( const std::string& p )
{
  d->project_id = p;
}

void vertex_ai_client::set_location( const std::string& l )
{
  d->location = l;
}

void vertex_ai_client::set_credentials_file( const std::string& f )
{
  d->credentials_file = f;
}

void vertex_ai_client::set_timeout_seconds( int s )
{
  d->timeout_seconds = s;
}

bool vertex_ai_client::authenticate()
{
  FILE* pipe = popen( "gcloud auth print-access-token 2>/dev/null", "r" );

  if( pipe )
  {
    char buffer[512];
    std::string token;

    while( fgets( buffer, sizeof( buffer ), pipe ) )
    {
      token += buffer;
    }

    int status = pclose( pipe );

    while( !token.empty() &&
           ( token.back() == '\n' || token.back() == '\r' ) )
    {
      token.pop_back();
    }

    if( status == 0 && !token.empty() )
    {
      d->access_token = token;
      LOG_INFO( d->logger, "Authenticated via gcloud CLI" );
      return true;
    }
  }

  // TODO: parse service account key file and do JWT->token exchange
  LOG_ERROR( d->logger, "Failed to obtain access token" );
  return false;
}

std::vector< prediction_result > vertex_ai_client::predict(
  const std::string& endpoint_id,
  const std::string& instances_json,
  const std::string& parameters_json )
{
  std::string path =
    "/v1/projects/" + d->project_id +
    "/locations/" + d->location +
    "/endpoints/" + endpoint_id + ":predict";

  std::string body = "{\"instances\":" + instances_json;
  if( !parameters_json.empty() )
  {
    body += ",\"parameters\":" + parameters_json;
  }
  body += "}";

  auto res = d->post_json( path, body );

  std::vector< prediction_result > results;

  if( res && res->status == 200 )
  {
    prediction_result r;
    r.raw_json = res->body;
    // TODO: parse "predictions" array from response JSON
    results.push_back( r );
  }
  else
  {
    LOG_ERROR( d->logger, "Predict failed: HTTP "
      << ( res ? std::to_string( res->status ) : "connection error" ) );
  }

  return results;
}

training_job vertex_ai_client::submit_training_job(
  const std::string& display_name,
  const std::string& container_image_uri,
  const std::vector< std::string >& args,
  const std::string& machine_type,
  const std::string& accelerator_type,
  int accelerator_count,
  const std::string& base_output_dir )
{
  std::string path =
    "/v1/projects/" + d->project_id +
    "/locations/" + d->location + "/customJobs";

  std::ostringstream body;
  body << "{\"displayName\":\"" << display_name << "\","
       << "\"jobSpec\":{\"workerPoolSpecs\":[{"
       << "\"machineSpec\":{\"machineType\":\"" << machine_type << "\"";

  if( !accelerator_type.empty() && accelerator_count > 0 )
  {
    body << ",\"acceleratorType\":\"" << accelerator_type << "\""
         << ",\"acceleratorCount\":" << accelerator_count;
  }

  body << "},\"replicaCount\":1,"
       << "\"containerSpec\":{\"imageUri\":\"" << container_image_uri << "\"";

  if( !args.empty() )
  {
    body << ",\"args\":[";
    for( size_t i = 0; i < args.size(); ++i )
    {
      if( i > 0 ) body << ",";
      body << "\"" << args[i] << "\"";
    }
    body << "]";
  }

  body << "}}]";

  if( !base_output_dir.empty() )
  {
    body << ",\"baseOutputDirectory\":{\"outputUriPrefix\":\""
         << base_output_dir << "\"}";
  }

  body << "}}";

  auto res = d->post_json( path, body.str() );

  training_job job;

  if( res && res->status == 200 )
  {
    job.raw_json = res->body;
    // TODO: parse "name" and "state" from response JSON
    LOG_INFO( d->logger, "Training job submitted" );
  }
  else
  {
    job.state = "SUBMIT_FAILED";
    job.error_message = res ? res->body : "connection error";
    LOG_ERROR( d->logger,
      "Training job submission failed: " << job.error_message );
  }

  return job;
}

training_job vertex_ai_client::get_job_status( const std::string& job_name )
{
  std::string path = "/v1/" + job_name;
  auto res = d->get_json( path );

  training_job job;
  job.job_name = job_name;

  if( res && res->status == 200 )
  {
    job.raw_json = res->body;
    // TODO: parse "state" from response JSON
  }

  return job;
}

training_job vertex_ai_client::wait_for_job(
  const std::string& job_name, int poll_interval_sec )
{
  while( true )
  {
    training_job job = get_job_status( job_name );

    if( job.state == "JOB_STATE_SUCCEEDED" ||
        job.state == "JOB_STATE_FAILED" ||
        job.state == "JOB_STATE_CANCELLED" )
    {
      return job;
    }

    LOG_INFO( d->logger, "Job " << job_name
      << " state: " << job.state
      << ", polling again in " << poll_interval_sec << "s" );

    std::this_thread::sleep_for(
      std::chrono::seconds( poll_interval_sec ) );
  }
}

} // namespace vertex_ai
} // namespace viame
