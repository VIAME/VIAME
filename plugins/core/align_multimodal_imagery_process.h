/*ckwg +29
 * Copyright 2019 by Kitware, Inc.
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
 * \brief Align multi modal images that may be temporally out of sync
 */

#ifndef VIAME_CORE_ALIGN_MULTIMODAL_IMAGERY_PROCESS_H
#define VIAME_CORE_ALIGN_MULTIMODAL_IMAGERY_PROCESS_H

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
 * Port names used by this process. Note: in the future these will be replaced
 * with generic names to increase the general usage of this process.
 */
create_port_trait( optical_image, image, "Optical image" );
create_port_trait( optical_timestamp, timestamp, "Optical timestamp" );
create_port_trait( optical_file_name, file_name, "Optical file name" );

create_port_trait( thermal_image, image, "Thermal image" );
create_port_trait( thermal_timestamp, timestamp, "Thermal timestamp" );
create_port_trait( thermal_file_name, file_name, "Thermal file name" );

create_port_trait( warped_optical_image, image, "Warped optical image" );
create_port_trait( warped_thermal_image, image, "Warped thermal image" );
create_port_trait( optical_to_thermal_homog, homography, "Homography" );
create_port_trait( thermal_to_optical_homog, homography, "Homography" );


// -----------------------------------------------------------------------------
/**
 * @brief Align multi modal images that may be temporally out of sync
 * 
 * It's also possible for derived versions of this class to attempt to
 * register the images together.
 */
class VIAME_PROCESSES_CORE_NO_EXPORT align_multimodal_imagery_process
  : public sprokit::process
{
public:
  // -- CONSTRUCTORS --
  align_multimodal_imagery_process( kwiver::vital::config_block_sptr const& config );
  virtual ~align_multimodal_imagery_process();

protected:
  virtual void _configure();
  virtual void _step();

  struct buffered_frame
  {
    buffered_frame( kwiver::vital::image_container_sptr _image,
                    kwiver::vital::timestamp _ts,
                    std::string _name )
     : image( _image ),
       ts( _ts ),
       name( _name )
    {}

    kwiver::vital::image_container_sptr image;
    kwiver::vital::timestamp ts;
    std::string name;

    double time()
    {
      return static_cast< double >( ts.get_time_usec() );
    }
  };

  virtual void attempt_registration( const buffered_frame& frame1,
                                     const buffered_frame& frame2,
                                     const bool output_frame1_time );

  virtual void output_no_match( const buffered_frame& frame,
                                const unsigned stream_id );

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;

}; // end class align_multimodal_imagery_process

} // end namespace core
} // end namespace viame

#endif // VIAME_CORE_ALIGN_MULTIMODAL_IMAGERY_PROCESS_H
