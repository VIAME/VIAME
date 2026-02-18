/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_VERTEX_AI_DETECTOR_H
#define VIAME_VERTEX_AI_DETECTOR_H

#include <sprokit/pipeline/process.h>

#include <memory>

namespace viame {
namespace vertex_ai {

/// Sprokit process that sends inference requests to a deployed Vertex AI
/// endpoint and returns results as detected_object_set or raw JSON.
class vertex_ai_detector
  : public sprokit::process
{
public:
  vertex_ai_detector(
    kwiver::vital::config_block_sptr const& config );
  virtual ~vertex_ai_detector();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr< priv > d;
};

} // namespace vertex_ai
} // namespace viame

#endif // VIAME_VERTEX_AI_DETECTOR_H
