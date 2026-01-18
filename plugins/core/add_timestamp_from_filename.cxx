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

namespace kv = kwiver::vital;
namespace kva = kv::algo;

// ----------------------------------------------------------------------------
add_timestamp_from_filename::add_timestamp_from_filename()
{
  this->set_capability( kva::image_io::HAS_TIME, true );
}

// ----------------------------------------------------------------------------
kv::config_block_sptr
  add_timestamp_from_filename::get_configuration() const
{
  auto config = kva::image_io::get_configuration();

  kv::get_nested_algo_configuration<kva::image_io>(
    "image_reader", config, this->image_reader);

  return config;
}

// ----------------------------------------------------------------------------
void add_timestamp_from_filename::set_configuration(
  kv::config_block_sptr config )
{
  auto new_config = this->get_configuration();
  new_config->merge_config( config );

  kv::set_nested_algo_configuration<kva::image_io>(
    "image_reader", new_config, this->image_reader );
}

// ----------------------------------------------------------------------------
bool add_timestamp_from_filename::check_configuration(
  kv::config_block_sptr config ) const
{
  return kv::check_nested_algo_configuration<kva::image_io>(
    "image_reader", config );
}

// ----------------------------------------------------------------------------
kv::image_container_sptr add_timestamp_from_filename::load_(
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
  kv::image_container_sptr data ) const
{
  if( this->image_reader )
  {
    this->image_reader->save( filename, data );
  }
}

// ----------------------------------------------------------------------------
kv::metadata_sptr add_timestamp_from_filename::load_metadata_(
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
kv::metadata_sptr add_timestamp_from_filename::fixup_metadata(
  std::string const& filename, kv::metadata_sptr md ) const
{
  if( !md )
  {
    md = std::make_shared<kv::metadata>();
  }

  kv::time_usec_t utc_time_usec = convert_to_timestamp( filename );

  kv::timestamp ts;
  ts.set_time_usec( utc_time_usec );

  md->set_timestamp( ts );
  return md;
}

}
