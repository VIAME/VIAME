/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Convert single image to image_set
 */

#ifndef VIAME_CORE_IMAGE_TO_IMAGE_SET_PROCESS_H
#define VIAME_CORE_IMAGE_TO_IMAGE_SET_PROCESS_H

#include <plugins/core/viame_processes_core_export.h>

#include <sprokit/pipeline/process.h>

#include <memory>

namespace viame
{

namespace core
{

// ----------------------------------------------------------------------------
/**
 * @brief Convert single image to image_set_sptr
 *
 * This process takes a single image and wraps it in an image_set_sptr
 * for use with processes that require image sets.
 */
class VIAME_PROCESSES_CORE_NO_EXPORT image_to_image_set_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "image_to_image_set",
               "Convert single image to image_set" )

  image_to_image_set_process( kwiver::vital::config_block_sptr const& config );
  virtual ~image_to_image_set_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;
};

} // end namespace core

} // end namespace viame

#endif // VIAME_CORE_IMAGE_TO_IMAGE_SET_PROCESS_H
