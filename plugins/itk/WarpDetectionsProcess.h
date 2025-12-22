/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Warp detections based on loaded ITK registration method
 */

#ifndef VIAME_ITK_WARP_DETECTIONS_PROCESS_H
#define VIAME_ITK_WARP_DETECTIONS_PROCESS_H

#include <sprokit/pipeline/process.h>

#include <plugins/itk/viame_itk_export.h>

#include <memory>

namespace viame
{

namespace itk
{

// -----------------------------------------------------------------------------
/**
 * @brief Filters out frames if certain criteria are not met
 */
class VIAME_ITK_NO_EXPORT itk_warp_detections_process
  : public sprokit::process
{
public:
  // -- CONSTRUCTORS --
  itk_warp_detections_process( kwiver::vital::config_block_sptr const& config );
  virtual ~itk_warp_detections_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;

}; // end class itk_warp_detections_process

} // end namespace itk
} // end namespace viame

#endif // VIAME_ITK_WARP_DETECTIONS_PROCESS_H
