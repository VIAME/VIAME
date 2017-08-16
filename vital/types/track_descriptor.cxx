/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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

#include "track_descriptor.h"

#include <sstream>


namespace kwiver {
namespace vital {

// Factory methods
track_descriptor_sptr
track_descriptor
::create( std::string const& type )
{
  track_descriptor_sptr output( new track_descriptor() );
  output->type_ = type;
  return output;
}


track_descriptor_sptr
track_descriptor
::create( track_descriptor_sptr to_copy )
{
  track_descriptor_sptr output( new track_descriptor() );
  *output = *to_copy;
  return output;
}


track_descriptor
::track_descriptor()
{
}


track_descriptor
::~track_descriptor()
{
}


void
track_descriptor
::set_type( const descriptor_id_t& type )
{
  this->type_ = type;
}


track_descriptor::descriptor_id_t const&
track_descriptor
::get_type() const
{
  return this->type_;
}


void
track_descriptor
::add_track_id( uint64_t id )
{
  this->track_ids_.push_back( id );
}


void
track_descriptor
::add_track_ids( const std::vector< uint64_t >& ids )
{
  this->track_ids_.insert( this->track_ids_.end(), ids.begin(), ids.end() );
}


std::vector< uint64_t > const&
track_descriptor
::get_track_ids() const
{
  return this->track_ids_;
}


void
track_descriptor
::set_descriptor( descriptor_data_sptr_t const& data )
{
  this->data_ = data;
}


track_descriptor::descriptor_data_sptr_t const&
track_descriptor
::get_descriptor() const
{
  return this->data_;
}


track_descriptor::descriptor_data_sptr_t&
track_descriptor
::get_descriptor()
{
  return this->data_;
}


double&
track_descriptor
::at( const size_t idx )
{
  // validate element index
  if ( idx >= this->data_->size() )
  {
    std::stringstream msg;
    msg << "Raw descriptor index " << idx
        << " is beyond the last feature element ("
        << this->data_->size() - 1 << ")";
    throw std::out_of_range( msg.str() );
  }

  return this->data_->raw_data()[idx];
}


double const&
track_descriptor
::at( const size_t idx ) const
{
  // validate element index
  if ( idx >= this->data_->size() )
  {
    std::stringstream msg;
    msg << "Raw descriptor index " << idx
        << " is beyond the last feature element ("
        << this->data_->size() - 1 << ")";
    throw std::out_of_range( msg.str() );
  }

  return this->data_->raw_data()[idx];
}


size_t
track_descriptor
::descriptor_size() const
{
  return this->data_->size();
}


void
track_descriptor
::resize_descriptor( size_t s )
{
  this->data_ = descriptor_data_sptr_t(
    new descriptor_data_t( s ) );
}


void
track_descriptor
::resize_descriptor( size_t s, double v )
{
  this->data_ = descriptor_data_sptr_t(
    new descriptor_data_t( s ) );

  std::fill( this->data_->raw_data(),
    this->data_->raw_data() + s, v );
}


bool
track_descriptor
::has_descriptor() const
{
  return this->data_->size() != 0;
}


void
track_descriptor
::set_history( descriptor_history_t const& hist )
{
  this->history_ = hist;
}


void
track_descriptor
::add_history_entry( track_descriptor::history_entry const& hist )
{
  this->history_.push_back( hist );
}


track_descriptor::descriptor_history_t const&
track_descriptor
::get_history() const
{
  return this->history_;
}


// ================================================================
track_descriptor::history_entry::
history_entry( const vital::timestamp& ts,
               const image_bbox_t& img_loc,
               const world_bbox_t& world_loc )
  : ts_(ts),
    img_loc_( img_loc ),
    world_loc_( world_loc )
{
}


track_descriptor::history_entry::
history_entry( const vital::timestamp& ts,
               const image_bbox_t& img_loc )
  : ts_( ts ),
    img_loc_( img_loc ),
    world_loc_( 0, 0, 0, 0 )
{
}


vital::timestamp
track_descriptor::history_entry::
get_timestamp() const
{
  return this->ts_;
}


track_descriptor::history_entry::image_bbox_t const&
track_descriptor::history_entry::
get_image_location() const
{
  return this->img_loc_;
}


track_descriptor::history_entry::world_bbox_t const&
track_descriptor::history_entry::
get_world_location() const
{
  return this->world_loc_;
}


} } // end namespace
