/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Calibrate a single camera from object track set
 */

#ifndef VIAME_OPENCV_CALIBRATE_SINGLE_CAMERA_PROCESS_H
#define VIAME_OPENCV_CALIBRATE_SINGLE_CAMERA_PROCESS_H

#include <sprokit/pipeline/process.h>

#include <plugins/opencv/viame_processes_opencv_export.h>

#include <memory>

namespace viame
{

// -----------------------------------------------------------------------------
/**
 * @brief Calibrate a single camera from object track set
 *
 * Performs monocular camera calibration using detected chessboard corners.
 * Outputs camera intrinsics in OpenCV YAML and JSON formats.
 */
class VIAME_PROCESSES_OPENCV_NO_EXPORT calibrate_single_camera_process
  : public sprokit::process
{
public:
  // -- CONSTRUCTORS --
  calibrate_single_camera_process( kwiver::vital::config_block_sptr const& config );
  virtual ~calibrate_single_camera_process();

protected:
  void _configure() override;
  void _step() override;

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;

}; // end class calibrate_single_camera_process

} // end namespace viame

#endif // VIAME_OPENCV_CALIBRATE_SINGLE_CAMERA_PROCESS_H
