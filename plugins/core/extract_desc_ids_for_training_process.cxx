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
 * \brief Extract descriptor IDs overlapping with groundtruth.
 */

#include "extract_desc_ids_for_training_process.h"

#include <vital/vital_types.h>

#include <vital/types/category_hierarchy.h>
#include <vital/types/timestamp_config.h>

#include <fstream>
#include <iostream>

#include <boost/filesystem.hpp>

namespace viame
{

namespace core
{

create_config_trait( category_file, std::string, "labels.txt",
  "Definition of semantic object types for training" );
create_config_trait( output_directory, std::string, "",
  "Output directory for descriptor ID lists" );
create_config_trait( output_extension, std::string, "lbl",
  "Extension of output descriptor ID files" );
create_config_trait( background_label, std::string, "background",
  "Label for output IDs which don't belong in any categories" );
create_config_trait( positive_min_overlap, double, "0.5",
  "Area overlap criteria for a detection and groundtruth box" );
create_config_trait( negative_max_overlap, double, "0.1",
  "Area overlap criteria for a background detection" );

//--------------------------------------------------------------------------------
// Private implementation class
class extract_desc_ids_for_training_process::priv
{
public:
  priv()
    : m_category_file( "labels.txt" )
    , m_output_directory( "" )
    , m_output_extension( "lbl" )
    , m_background_label( "background" )
    , m_positive_min_overlap( 0.5 )
    , m_negative_max_overlap( 0.1 ) {}

  ~priv() {}

  std::string m_category_file;
  std::string m_output_directory;

  std::string m_output_extension;
  std::string m_background_label;

  double m_positive_min_overlap;
  double m_negative_max_overlap;

  kwiver::vital::category_hierarchy_sptr m_classes;
  std::vector< std::unique_ptr< std::ofstream > > m_writers;
};

// ===============================================================================

extract_desc_ids_for_training_process
::extract_desc_ids_for_training_process( config_block_sptr const& config )
  : process( config ),
    d( new extract_desc_ids_for_training_process::priv() )
{
  make_ports();
  make_config();
}


extract_desc_ids_for_training_process
::~extract_desc_ids_for_training_process()
{
  for( unsigned i = 0; i < d->m_writers.size(); ++i )
  {
    if( d->m_writers[i] && d->m_writers[i]->is_open() )
    {
      d->m_writers[i]->close();
    }
  }
}


// -------------------------------------------------------------------------------
void
extract_desc_ids_for_training_process
::_configure()
{
  d->m_category_file = config_value_using_trait( category_file );
  d->m_output_directory = config_value_using_trait( output_directory );
  d->m_output_extension = config_value_using_trait( output_extension );
  d->m_background_label = config_value_using_trait( background_label );
  d->m_positive_min_overlap = config_value_using_trait( positive_min_overlap );
  d->m_negative_max_overlap = config_value_using_trait( negative_max_overlap );

  d->m_classes.reset( new kwiver::vital::category_hierarchy( d->m_category_file ) );
  d->m_writers.resize( d->m_classes->size() + 1 );

  std::string filename = d->m_background_label + "." + d->m_output_extension;

  if( !d->m_output_directory.empty() )
  {
    filename = d->m_output_directory + "/" + filename;

    if( !boost::filesystem::exists( d->m_output_directory ) )
    {
      boost::filesystem::create_directories( d->m_output_directory );
    }
  }

  d->m_writers[ d->m_classes->size() ] = std::unique_ptr< std::ofstream >(
    new std::ofstream( filename, std::ofstream::out | std::ofstream::app ) );

  for( auto label : d->m_classes->all_class_names() )
  {
    filename = label + "." + d->m_output_extension;

    if( !d->m_output_directory.empty() )
    {
      filename = d->m_output_directory + "/" + filename;
    }

    unsigned index = d->m_classes->get_class_id( label );

    d->m_writers[ index ] = std::unique_ptr< std::ofstream >(
      new std::ofstream( filename, std::ofstream::out | std::ofstream::app ) );
  }
}


// -------------------------------------------------------------------------------
void
extract_desc_ids_for_training_process
::_step()
{
  bool timestamp_set = false;
  kwiver::vital::timestamp timestamp;

  kwiver::vital::track_descriptor_set_sptr descriptors;
  kwiver::vital::detected_object_set_sptr detections;

  if( has_input_port_edge_using_trait( timestamp ) )
  {
    timestamp_set = true;
    timestamp = grab_from_port_using_trait( timestamp );
  }

  descriptors = grab_from_port_using_trait( track_descriptor_set );
  detections = grab_from_port_using_trait( detected_object_set );

  for( kwiver::vital::track_descriptor_sptr desc : *descriptors )
  {
    // Find bounding box for current frame
    kwiver::vital::bounding_box_d desc_box( 0, 0, 0, 0 );

    if( !timestamp_set && desc->get_history().size() == 1 )
    {
      desc_box = desc->get_history()[0].get_image_location();
    }
    else
    {
      for( const auto& hist_entry : desc->get_history() )
      {
        if( hist_entry.get_timestamp() == timestamp )
        {
          desc_box = hist_entry.get_image_location();
          break;
        }
      }
    }

    if( desc_box.width() == 0 || desc_box.height() == 0 )
    {
      continue;
    }

    bool is_background = true;

    for( kwiver::vital::detected_object_sptr det : *detections )
    {
      // Check type on detection, is it in our training set
      kwiver::vital::class_map_sptr type_sptr = det->type();

      std::string top_category;
      double top_score;

      type_sptr->get_most_likely( top_category, top_score );

      if( !d->m_classes->has_class_name( top_category ) )
      {
        continue;
      }

      // Check bounding box overlap with detection
      const kwiver::vital::bounding_box_d& det_box =
        det->bounding_box();

      kwiver::vital::bounding_box_d intersect =
        kwiver::vital::intersection( desc_box, det_box );

      // Print out in correct category file if match
      if( intersect.height() <= 0 || intersect.width() <= 0 )
      {
        continue;
      }
      else
      {
        double min_overlap = std::min(
          static_cast< double >( intersect.area() ) / det_box.area(), 
          static_cast< double >( intersect.area() ) / desc_box.area() );

        if( min_overlap >= d->m_positive_min_overlap )
        {
          *d->m_writers[ d->m_classes->get_class_id( top_category ) ]
            << desc->get_uid().value() << std::endl;

          is_background = false;
        }
        else if( min_overlap >= d->m_negative_max_overlap )
        {
          is_background = false;
        }
      }
    }

    if( is_background )
    {
      // Print out in correct category file if match
      *d->m_writers.back() << desc->get_uid().value() << std::endl;
    }
  }
}


// -------------------------------------------------------------------------------
void
extract_desc_ids_for_training_process
::make_ports()
{
  sprokit::process::port_flags_t optional;

  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  // -- inputs --
  declare_input_port_using_trait( timestamp, optional );

  declare_input_port_using_trait( track_descriptor_set, required );
  declare_input_port_using_trait( detected_object_set, required );
}


// -------------------------------------------------------------------------------
void
extract_desc_ids_for_training_process
::make_config()
{
  declare_config_using_trait( category_file );
  declare_config_using_trait( output_directory );
  declare_config_using_trait( output_extension );
  declare_config_using_trait( background_label );
  declare_config_using_trait( positive_min_overlap );
  declare_config_using_trait( negative_max_overlap );
}

} // end namespace core

} // end namespace viame
