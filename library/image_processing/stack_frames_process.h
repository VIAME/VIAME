/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Stack multiple frames together onto the same output
 */

#ifndef VIAME_IMAGE_PROCESSING_STACK_FRAMES_PROCESS_H
#define VIAME_IMAGE_PROCESSING_STACK_FRAMES_PROCESS_H

#include <viame/pipeline_framework/process.h>

#include "viame_processes_image_processing_export.h"

#include <viame/pipeline_framework/type_traits.h>

#include <viame/core_types/image_container.h>
#include <viame/core_types/timestamp.h>

#include <memory>

namespace viame
{

namespace core
{

// -----------------------------------------------------------------------------
/**
 * @brief Stack frames with some gap into one output image temporally.
 */
class VIAME_PROCESSES_IMAGE_PROCESSING_EXPORT stack_frames_process
  : public viame::pipeline::process
{
public:
  // -- CONSTRUCTORS --
  stack_frames_process( viame::config_block_sptr const& config );
  virtual ~stack_frames_process();

protected:
  virtual void _configure();
  virtual void _step();

  struct buffered_frame
  {
    buffered_frame( viame::image_container_sptr _image,
                    viame::timestamp _ts )
     : image( _image ),
       ts( _ts )
    {}

    viame::image_container_sptr image;
    viame::timestamp ts;

    double time()
    {
      return static_cast< double >( ts.get_time_usec() );
    }
  };

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;

}; // end class stack_frames_process

} // end namespace core
} // end namespace viame

#endif // VIAME_IMAGE_PROCESSING_STACK_FRAMES_PROCESS_H
