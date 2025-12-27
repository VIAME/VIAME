/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Stack multiple frames together onto the same output
 */

#ifndef VIAME_CORE_STACK_FRAMES_PROCESS_H
#define VIAME_CORE_STACK_FRAMES_PROCESS_H

#include <sprokit/pipeline/process.h>

#include <plugins/core/viame_processes_core_export.h>

#include <sprokit/processes/kwiver_type_traits.h>

#include <vital/types/image_container.h>
#include <vital/types/timestamp.h>

#include <memory>

namespace viame
{

namespace core
{

// -----------------------------------------------------------------------------
/**
 * @brief Stack frames with some gap into one output image temporally.
 */
class VIAME_PROCESSES_CORE_NO_EXPORT stack_frames_process
  : public sprokit::process
{
public:
  // -- CONSTRUCTORS --
  stack_frames_process( kwiver::vital::config_block_sptr const& config );
  virtual ~stack_frames_process();

protected:
  virtual void _configure();
  virtual void _step();

  struct buffered_frame
  {
    buffered_frame( kwiver::vital::image_container_sptr _image,
                    kwiver::vital::timestamp _ts )
     : image( _image ),
       ts( _ts )
    {}

    kwiver::vital::image_container_sptr image;
    kwiver::vital::timestamp ts;

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

#endif // VIAME_CORE_STACK_FRAMES_PROCESS_H
