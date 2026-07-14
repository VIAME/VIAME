// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "merge_detections_suppress_in_regions.h"

#include <cctype>
#include <string>
#include <algorithm>

namespace viame {

namespace kv = kwiver::vital;

/// Helper Function
bool
merge_detections_suppress_in_regions
::compare_classes( const std::string& c1, const std::string& c2 ) const
{
  if( c_case_sensitive )
  {
    return c1 == c2;
  }

  return std::equal( c1.begin(), c1.end(), c2.begin(), c2.end(),
    []( const unsigned char& i, const unsigned char& j )
    {
      return std::tolower( i ) == std::tolower( j );
    } );
}

/// Check that the algorithm's currently configuration is valid
bool
merge_detections_suppress_in_regions
::check_configuration( VITAL_UNUSED kv::config_block_sptr config ) const
{
  return true;
}

// -----------------------------------------------------------------------------
kv::detected_object_set_sptr
merge_detections_suppress_in_regions
::merge( std::vector< kv::detected_object_set_sptr > const& sets ) const
{
  // Returns detections sorted by confidence threshold
  kv::detected_object_set_sptr output( new kv::detected_object_set() );

  if( sets.empty() || !sets[0] )
  {
    return output;
  }
  if( sets.size() == 1 )
  {
    return sets[0];
  }

  kv::detected_object_set_sptr region_set = sets[0];

  for( unsigned i = 1; i < sets.size(); i++ )
  {
    auto test_set = sets[i];

    for( auto det : *test_set )
    {
      bool should_add = true, should_adjust = false;
      const auto det_bbox = det->bounding_box();

      for( auto region : *region_set )
      {
        const auto reg_bbox = region->bounding_box();
        const auto overlap = kv::intersection( det_bbox, reg_bbox );

        // Check how much they overlap. Only keep if the overlapped percent isn't too high
        if( overlap.min_x() < overlap.max_x() && overlap.min_y() < overlap.max_y() &&
            ( overlap.area() / det_bbox.area() ) >= c_min_overlap )
        {
          std::string reg_class;

          if( region->type() )
          {
            region->type()->get_most_likely( reg_class );
          }

          if( compare_classes( c_suppression_class, reg_class ) ||
              ( c_suppression_class.empty() && c_borderline_class.empty() ) )
          {
            should_add = false;
          }
          else if( !c_borderline_class.empty() &&
                    compare_classes( c_borderline_class, reg_class ) )
          {
            should_adjust = true;
          }
        }
      }
      if( should_add ) // It doesn't overlap too much, add it in
      {
        if( should_adjust )
        {
          auto new_det = det->clone();
          auto adj_type = new_det->type();

          for( auto cls_pair : *adj_type )
          {
            cls_pair.second = cls_pair.second * c_borderline_scale_factor;
          }

          new_det->set_type( adj_type );
          output->add( new_det );
        }
        else
        {
          output->add( det );
        }
      }
    }
  }

  if( c_output_region_classes )
  {
    for( auto region : *region_set )
    {
      output->add( region );
    }
  }

  return output;
}

} // end namespace viame
