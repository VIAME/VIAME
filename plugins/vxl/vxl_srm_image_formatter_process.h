/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Format images in a way optimized for later IQR processing
 */

#ifndef VIAME_VXL_SRM_IMAGE_FORMATTER_PROCESS_H
#define VIAME_VXL_SRM_IMAGE_FORMATTER_PROCESS_H

#include <sprokit/pipeline/process.h>

#include <plugins/vxl/viame_processes_vxl_export.h>

#include <sprokit/processes/kwiver_type_traits.h>

#include <vital/types/image_container.h>
#include <vital/types/timestamp.h>

#include <memory>

namespace viame
{

namespace vxl
{

// -----------------------------------------------------------------------------
/**
 * @brief Format images in a way optimized for later IQR processing
 * 
 * Depending on parameters this operation could be to perform image resizing,
 * perform large image tiling, or other operations. It exists mostly due to the
 * way the current descktop GUI interface uses video-based KWA files for image
 * storage, to reduce disk usage and increase processing speeds.
 */
class VIAME_PROCESSES_VXL_NO_EXPORT vxl_srm_image_formatter_process
  : public sprokit::process
{
public:
  // -- CONSTRUCTORS --
  vxl_srm_image_formatter_process( kwiver::vital::config_block_sptr const& config );
  virtual ~vxl_srm_image_formatter_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;

}; // end class vxl_srm_image_formatter_process

} // end namespace vxl
} // end namespace viame

#endif // VIAME_VXL_SRM_IMAGE_FORMATTER_PROCESS_H
