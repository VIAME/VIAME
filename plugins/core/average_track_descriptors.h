/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_CORE_AVERAGE_TRACK_DESCRIPTORS_H
#define VIAME_CORE_AVERAGE_TRACK_DESCRIPTORS_H

#include "viame_core_export.h"

#include <vital/algo/compute_track_descriptors.h>

#include <memory>

namespace viame {

class VIAME_CORE_EXPORT average_track_descriptors
  : public kwiver::vital::algorithm_impl< average_track_descriptors,
      kwiver::vital::algo::compute_track_descriptors >
{
public:
  PLUGIN_INFO( "average",
               "Track descriptor consolidation using simple averaging" )

  average_track_descriptors();
  virtual ~average_track_descriptors();

  virtual kwiver::vital::config_block_sptr get_configuration() const;
  virtual void set_configuration( kwiver::vital::config_block_sptr config );
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual kwiver::vital::track_descriptor_set_sptr
  compute( kwiver::vital::timestamp ts,
           kwiver::vital::image_container_sptr image_data,
           kwiver::vital::object_track_set_sptr tracks );

  virtual kwiver::vital::track_descriptor_set_sptr flush();

private:
  class priv;
  const std::unique_ptr<priv> d;
};

} // end namespace viame

#endif // VIAME_CORE_AVERAGE_TRACK_DESCRIPTORS_H
