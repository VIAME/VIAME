/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "add_timestamp_from_filename.h"

#include <vital/algo/algorithm.txx>

#include "filename_to_timestamp.h"

#include <kwiversys/SystemTools.hxx>

#include <sstream>
#include <string>
#include <ctime>
#include <regex>

#ifdef WIN32
#define timegm _mkgmtime
#endif

namespace viame
{

// ----------------------------------------------------------------------------
add_timestamp_from_filename::add_timestamp_from_filename()
{
  this->set_capability( kwiver::vital::algo::image_io::HAS_TIME, true );
}

// ----------------------------------------------------------------------------
kwiver::vital::config_block_sptr
  add_timestamp_from_filename::get_configuration() const
{
  auto config = kwiver::vital::algo::image_io::get_configuration();

  kwiver::vital::get_nested_algo_configuration<kwiver::vital::algo::image_io>(
    "image_reader", config, this->image_reader);

  return config;
}

// ----------------------------------------------------------------------------
void add_timestamp_from_filename::set_configuration(
  kwiver::vital::config_block_sptr config )
{
  auto new_config = this->get_configuration();
  new_config->merge_config( config );

  kwiver::vital::set_nested_algo_configuration<kwiver::vital::algo::image_io>(
    "image_reader", new_config, this->image_reader );
}

// ----------------------------------------------------------------------------
bool add_timestamp_from_filename::check_configuration(
  kwiver::vital::config_block_sptr config ) const
{
  return kwiver::vital::check_nested_algo_configuration<kwiver::vital::algo::image_io>(
    "image_reader", config );
}

// ----------------------------------------------------------------------------
kwiver::vital::image_container_sptr add_timestamp_from_filename::load_(
  std::string const& filename ) const
{
  if( this->image_reader )
  {
    auto im = this->image_reader->load( filename );
    im->set_metadata( this->fixup_metadata( filename, im->get_metadata() ) );
    return im;
  }

  return nullptr;
}

// ----------------------------------------------------------------------------
void add_timestamp_from_filename::save_(
  std::string const& filename,
  kwiver::vital::image_container_sptr data ) const
{
  if( this->image_reader )
  {
    this->image_reader->save( filename, data );
  }
}

// ----------------------------------------------------------------------------
kwiver::vital::metadata_sptr add_timestamp_from_filename::load_metadata_(
  std::string const& filename) const
{
  if( this->image_reader )
  {
    return this->fixup_metadata( filename,
      this->image_reader->load_metadata( filename ) );
  }

  return this->fixup_metadata( filename, nullptr );
}

// ----------------------------------------------------------------------------
kwiver::vital::metadata_sptr add_timestamp_from_filename::fixup_metadata(
  std::string const& filename, kwiver::vital::metadata_sptr md ) const
{
  if( !md )
  {
    md = std::make_shared<kwiver::vital::metadata>();
  }

  kwiver::vital::time_usec_t utc_time_usec = convert_to_timestamp( filename );

  kwiver::vital::timestamp ts;
  ts.set_time_usec( utc_time_usec );

  md->set_timestamp( ts );
  return md;
}

}
