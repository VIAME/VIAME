/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Accumulate image statistics (frame count, image dimensions) over a stream
 */

#ifndef VIAME_CORE_ACCUMULATE_IMAGE_STATISTICS_PROCESS_H
#define VIAME_CORE_ACCUMULATE_IMAGE_STATISTICS_PROCESS_H

#include <sprokit/pipeline/process.h>

#include <plugins/core/viame_processes_core_export.h>

#include <memory>

namespace viame
{

namespace core
{

// -----------------------------------------------------------------------------
/**
 * @brief Accumulate image statistics over a stream of images
 *
 * This process counts the number of input frames and tracks their dimensions.
 * It outputs the total frame count and image width/height once all inputs
 * have been processed.
 */
class VIAME_PROCESSES_CORE_NO_EXPORT accumulate_image_statistics_process
  : public sprokit::process
{
public:
  // -- CONSTRUCTORS --
  accumulate_image_statistics_process( kwiver::vital::config_block_sptr const& config );
  virtual ~accumulate_image_statistics_process();

protected:
  void _configure() override;
  void _step() override;

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;

}; // end class accumulate_image_statistics_process

} // end namespace core
} // end namespace viame

#endif // VIAME_CORE_ACCUMULATE_IMAGE_STATISTICS_PROCESS_H
