// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "merge_detections_suppress_in_regions.h"

#include <cctype>
#include <string>
#include <algorithm>

namespace viame {

namespace kv = kwiver::vital;

/// Private implementation class
class merge_detections_suppress_in_regions::priv
{
public:

  /// Constructor
  priv()
  : suppression_class( "" ),
    borderline_class( "" ),
    borderline_scale_factor( 0.5 ),
    min_overlap( 0.5 ),
    output_region_classes( true ),
    case_sensitive( false )
  {}

  /// Destructor
  ~priv() {}

  /// Parameters
  std::string suppression_class;
  std::string borderline_class;
  double borderline_scale_factor;
  double min_overlap;
  bool output_region_classes;
  bool case_sensitive;

  /// Functions
  bool compare_classes( const std::string& c1, const std::string& c2 );
};


/// Helper Function
bool
merge_detections_suppress_in_regions::priv
::compare_classes( const std::string& c1, const std::string& c2 )
{
  if( case_sensitive )
  {
    return c1 == c2;
  }

  return std::equal( c1.begin(), c1.end(), c2.begin(), c2.end(),
    []( const unsigned char& i, const unsigned char& j )
    {
      return std::tolower( i ) == std::tolower( j );
    } );
}


/// Constructor
merge_detections_suppress_in_regions
::merge_detections_suppress_in_regions()
: d( new priv() )
{
}


/// Destructor
merge_detections_suppress_in_regions
::~merge_detections_suppress_in_regions()
{
}


/// Get this algorithm's \link vital::config_block configuration block \endlink
kv::config_block_sptr
merge_detections_suppress_in_regions
::get_configuration() const
{
  kv::config_block_sptr config = kv::algo::merge_detections::get_configuration();

  config->set_value( "suppression_class", d->suppression_class,
    "Suppression region class IDs, will eliminate any detections overlapping with "
    "this class entirely." );

  config->set_value( "borderline_class", d->borderline_class,
    "Borderline region class IDs, will reduce the probability of any detections "
    "overlapping with the class by some fixed scale factor." );

  config->set_value( "borderline_scale_factor", d->borderline_scale_factor,
    "The factor by which the detections are scaled when overlapping with borderline "
    "regions." );

  config->set_value( "min_overlap", d->min_overlap,
    "The minimum percent a detection can overlap with a suppression category before "
    "it's discarded or reduced. Range [0.0,1.0]." );

  config->set_value( "output_region_classes", d->output_region_classes,
    "Add suppression and borderline classes to output." );

  config->set_value( "case_sensitive", d->case_sensitive,
    "Treat class names as case sensitive or insensitive." );

  return config;
}


/// Set this algorithm's properties via a config block
void
merge_detections_suppress_in_regions
::set_configuration( kv::config_block_sptr in_config )
{
  kv::config_block_sptr config = this->get_configuration();
  config->merge_config( in_config );

  d->suppression_class = config->get_value< std::string >( "suppression_class" );
  d->borderline_class = config->get_value< std::string >( "borderline_class" );
  d->borderline_scale_factor = config->get_value< double >( "borderline_scale_factor" );
  d->min_overlap = config->get_value< double >( "min_overlap" );
  d->output_region_classes = config->get_value< bool >( "output_region_classes" );
  d->case_sensitive = config->get_value< bool >( "case_sensitive" );
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
            ( overlap.area() / det_bbox.area() ) >= d->min_overlap )
        {
          std::string reg_class;

          if( region->type() )
          {
            region->type()->get_most_likely( reg_class );
          }

          if( d->compare_classes( d->suppression_class, reg_class ) ||
              ( d->suppression_class.empty() && d->borderline_class.empty() ) )
          {
            should_add = false;
          }
          else if( !d->borderline_class.empty() &&
                    d->compare_classes( d->borderline_class, reg_class ) )
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
            cls_pair.second = cls_pair.second * d->borderline_scale_factor;
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

  if( d->output_region_classes )
  {
    for( auto region : *region_set )
    {
      output->add( region );
    }
  }

  return output;
}

} // end namespace viame
