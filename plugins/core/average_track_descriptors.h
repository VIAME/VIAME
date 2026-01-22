/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_CORE_AVERAGE_TRACK_DESCRIPTORS_H
#define VIAME_CORE_AVERAGE_TRACK_DESCRIPTORS_H

#include "viame_core_export.h"

#include <vital/algo/compute_track_descriptors.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

#include <deque>
#include <map>
#include <vector>

namespace viame {

class VIAME_CORE_EXPORT average_track_descriptors
  : public kwiver::vital::algo::compute_track_descriptors
{
public:
  PLUGGABLE_IMPL(
    average_track_descriptors,
    "Track descriptor consolidation using simple averaging",
    PARAM_DEFAULT(
      rolling, bool,
      "When set, produce an output for each input as the rolling average "
      "of the last N descriptors, where N is the interval. When reset, "
      "produce an output only for the first input and then every Nth input "
      "thereafter for any given track.",
      false ),
    PARAM_DEFAULT(
      interval, unsigned int,
      "When the interval is N, every descriptor output (after the first N inputs) "
      "is based on the last N descriptors seen as input for the given track.",
      5 )
  )

  virtual ~average_track_descriptors() = default;

  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual kwiver::vital::track_descriptor_set_sptr
  compute( kwiver::vital::timestamp ts,
           kwiver::vital::image_container_sptr image_data,
           kwiver::vital::object_track_set_sptr tracks );

  virtual kwiver::vital::track_descriptor_set_sptr flush();

private:
  void initialize() override;

  kwiver::vital::logger_handle_t m_logger;
  std::map< kwiver::vital::track_id_t, std::deque< std::vector< double > > > m_history;
};

} // end namespace viame

#endif // VIAME_CORE_AVERAGE_TRACK_DESCRIPTORS_H
