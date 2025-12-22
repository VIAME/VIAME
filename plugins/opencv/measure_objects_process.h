/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Stereo measurement process using calibration data
 */

#ifndef VIAME_OPENCV_MEASURE_OBJECTS_PROCESS_H
#define VIAME_OPENCV_MEASURE_OBJECTS_PROCESS_H

#include <sprokit/pipeline/process.h>

#include <plugins/opencv/viame_opencv_export.h>

#include <memory>

namespace viame
{

// -----------------------------------------------------------------------------
/**
 * @brief Stereo measurement process that matches detections between left and
 *        right cameras and computes fish length measurements using triangulation.
 *
 * This process takes detection sets from two stereo cameras, matches detections
 * between them based on reprojection error, triangulates matched keypoints,
 * and computes length measurements. Output includes detection sets with length
 * annotations and track sets for matched/unmatched detections.
 */
class VIAME_OPENCV_NO_EXPORT measure_objects_process
  : public sprokit::process
{
public:
  // -- CONSTRUCTORS --
  measure_objects_process( kwiver::vital::config_block_sptr const& config );
  virtual ~measure_objects_process();

protected:
  void _configure() override;
  void _step() override;

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;

}; // end class measure_objects_process

} // end namespace viame

#endif // VIAME_OPENCV_MEASURE_OBJECTS_PROCESS_H
