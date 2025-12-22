/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

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
