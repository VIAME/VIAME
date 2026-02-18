/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_VERTEX_AI_CLIENT_H
#define VIAME_VERTEX_AI_CLIENT_H

#include <string>
#include <vector>
#include <memory>

#include <vital/logger/logger.h>

namespace viame {
namespace vertex_ai {

struct prediction_result
{
  std::vector< double > scores;
  std::vector< std::string > labels;
  std::string raw_json;
};

struct training_job
{
  std::string job_name;
  std::string state;
  std::string error_message;
  std::string raw_json;
};

/// REST client for Google Cloud Vertex AI
class vertex_ai_client
{
public:
  vertex_ai_client();
  ~vertex_ai_client();

  void set_project( const std::string& project_id );
  void set_location( const std::string& location );
  void set_credentials_file( const std::string& path );
  void set_timeout_seconds( int seconds );

  /// Obtain OAuth2 access token (from gcloud CLI or service-account key)
  bool authenticate();

  /// POST to a deployed endpoint
  std::vector< prediction_result > predict(
    const std::string& endpoint_id,
    const std::string& instances_json,
    const std::string& parameters_json = "" );

  /// Submit a custom training job
  training_job submit_training_job(
    const std::string& display_name,
    const std::string& container_image_uri,
    const std::vector< std::string >& args,
    const std::string& machine_type,
    const std::string& accelerator_type = "",
    int accelerator_count = 0,
    const std::string& base_output_dir = "" );

  /// Poll job state
  training_job get_job_status( const std::string& job_name );

  /// Block until job finishes (poll loop with backoff)
  training_job wait_for_job( const std::string& job_name,
                             int poll_interval_sec = 30 );

private:
  class priv;
  std::unique_ptr< priv > d;
};

} // namespace vertex_ai
} // namespace viame

#endif // VIAME_VERTEX_AI_CLIENT_H
