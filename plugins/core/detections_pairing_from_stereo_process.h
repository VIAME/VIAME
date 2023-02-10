/**
 * \file
 * \brief Compute object detections pair from stereo depth map information
 */

#ifndef VIAME_DETECTIONS_PAIRING_FROM_STEREO_PROCESS_H
#define VIAME_DETECTIONS_PAIRING_FROM_STEREO_PROCESS_H

#include <sprokit/pipeline/process.h>

#include <plugins/core/viame_processes_core_export.h>

#include <memory>

namespace viame
{

namespace core
{

class detections_pairing_from_stereo;

// -----------------------------------------------------------------------------
/**
 * @brief Compute object detection pairs from stereo depth map information
 */
class VIAME_PROCESSES_CORE_NO_EXPORT detections_pairing_from_stereo_process
  : public sprokit::process
{
public:
  // -- CONSTRUCTORS --
  detections_pairing_from_stereo_process( kwiver::vital::config_block_sptr const& config );
  virtual ~detections_pairing_from_stereo_process();

protected:
  void _configure() override;
  void _step() override;

private:
  void make_ports();
  void make_config();

  const std::unique_ptr<detections_pairing_from_stereo> d;

};
} // core
} // viame

#endif // VIAME_DETECTIONS_PAIRING_FROM_STEREO_PROCESS_H
