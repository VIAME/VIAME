/**
 * \file
 * \brief Compute object detections pair from stereo depth map information
 */

#ifndef VIAME_OCV_PAIR_STEREO_DETECTIONS_PROCESS_H
#define VIAME_OCV_PAIR_STEREO_DETECTIONS_PROCESS_H

#include <sprokit/pipeline/process.h>

#include <plugins/opencv/viame_processes_opencv_export.h>

#include <memory>

namespace viame
{

class ocv_pair_stereo_detections;

// -----------------------------------------------------------------------------
/**
 * @brief Compute object detection pairs from stereo depth map information
 */
class VIAME_PROCESSES_OPENCV_NO_EXPORT ocv_pair_stereo_detections_process
  : public sprokit::process
{
public:
  // -- CONSTRUCTORS --
  ocv_pair_stereo_detections_process( kwiver::vital::config_block_sptr const& config );
  virtual ~ocv_pair_stereo_detections_process();

protected:
  void _configure() override;
  void _step() override;

private:
  void make_ports();
  void make_config();

  const std::unique_ptr<ocv_pair_stereo_detections> d;

};

} // viame

#endif // VIAME_OCV_PAIR_STEREO_DETECTIONS_PROCESS_H
