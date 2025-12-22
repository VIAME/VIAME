/*ckwg +29
 * Copyright 2022 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * \file
 * \brief Stack multiple frames together onto the same output
 */

#ifndef VIAME_CORE_FRAME_STACKER_PROCESS_H
#define VIAME_CORE_FRAME_STACKER_PROCESS_H

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
class VIAME_PROCESSES_CORE_NO_EXPORT frame_stacker_process
  : public sprokit::process
{
public:
  // -- CONSTRUCTORS --
  frame_stacker_process( kwiver::vital::config_block_sptr const& config );
  virtual ~frame_stacker_process();

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

}; // end class frame_stacker_process

} // end namespace core
} // end namespace viame

#endif // VIAME_CORE_FRAME_STACKER_PROCESS_H
