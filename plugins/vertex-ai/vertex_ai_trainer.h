/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_VERTEX_AI_TRAINER_H
#define VIAME_VERTEX_AI_TRAINER_H

#include <sprokit/pipeline/process.h>

#include <memory>

namespace viame {
namespace vertex_ai {

/// Sprokit process that submits a custom training job to Vertex AI
/// and optionally waits for completion. Runs once per pipeline execution.
class vertex_ai_trainer
  : public sprokit::process
{
public:
  vertex_ai_trainer(
    kwiver::vital::config_block_sptr const& config );
  virtual ~vertex_ai_trainer();

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

#endif // VIAME_VERTEX_AI_TRAINER_H
