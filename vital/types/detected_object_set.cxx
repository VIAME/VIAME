/*ckwg +29
 * Copyright 2016-2018 by Kitware, Inc.
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

#include "detected_object_set.h"
#include "bounding_box.h"


#include <algorithm>
#include <stdexcept>

namespace kwiver {
namespace vital {

// ==================================================================
namespace {

struct descending_confidence
{
  bool operator()( detected_object_sptr const& a, detected_object_sptr const& b ) const
  {
    if( a && !b )
    {
      return true;
    }
    else if( !a )
    {
      return false;
    }
    return a->confidence() > b->confidence();
  }
};

template < typename T1, typename T2 >
struct more_first
{
  typedef std::pair< T1, T2 > type;
  bool operator()( type const& a, type const& b ) const
  {
    return a.first > b.first;
  }
};

} // end namespace


// ------------------------------------------------------------------
detected_object_set::
detected_object_set()
{ }


// ------------------------------------------------------------------
detected_object_set::
detected_object_set( std::vector< detected_object_sptr > const& objs )
  : m_detected_objects( objs )
{
}


// ------------------------------------------------------------------
detected_object_set_sptr
detected_object_set::
clone() const
{
  auto new_obj = std::make_shared<detected_object_set>();

  auto ie = cend();
  for ( auto ix = cbegin(); ix != ie; ++ix )
  {
    // copy detection
    new_obj->add( (*ix)->clone() );
  }

  // duplicate attributes
  if ( this->m_attrs )
  {
    new_obj->m_attrs = this->m_attrs->clone();
  }

  return new_obj;
}


// ------------------------------------------------------------------
void
detected_object_set::
add( detected_object_sptr object )
{
  if ( ! object )
  {
    throw std::runtime_error( "Passing null pointer to detected_object_set::add()" );
  }

  m_detected_objects.push_back( object );
}


// ------------------------------------------------------------------
void
detected_object_set::
add( detected_object_set_sptr detections )
{
  for ( auto dptr : *detections )
  {
    this->add( dptr );
  }
}


// ------------------------------------------------------------------
size_t
detected_object_set::
size() const
{
  return m_detected_objects.size();
}


// ------------------------------------------------------------------
bool
detected_object_set::
empty() const
{
  return m_detected_objects.empty();
}


// ------------------------------------------------------------------
detected_object_set_sptr
detected_object_set::
select( double threshold ) const
{
  // The main list can get out of order if somebody updates the
  // confidence value of a detection directly
  std::vector< detected_object_sptr> vect;

  auto ie =  cend();
  for ( auto ix = cbegin(); ix != ie; ++ix )
  {
    if ( (*ix)->confidence() >= threshold )
    {
      vect.push_back( *ix );
    }
  }

  std::sort( vect.begin(), vect.end(), descending_confidence() );
  return std::make_shared< detected_object_set > (vect);
}


// ------------------------------------------------------------------
detected_object_set_sptr
detected_object_set::
select( const std::string& class_name, double threshold )const
{
  // Intermediate sortable data structure
  std::vector< std::pair< double, detected_object_sptr > > data;

  // Create a sortable list by selecting
  auto ie = cend();
  for ( auto ix = cbegin(); ix != ie; ++ix )
  {
    auto obj_type = (*ix)->type();
    if ( ! obj_type )
    {
      continue;  // Must have a type assigned
    }

    double score(0);
    try
    {
      score = obj_type->score( class_name );
    }
    catch (const std::runtime_error& )
    {
      // Object did not have the desired class_name. This not fatal,
      // but since we are looking for that name, there is some
      // expectation that it is present.

      //+ maybe log something?
      continue;
    }

    // Select those not below threshold
    if ( score >= threshold )
    {
      data.push_back( std::pair< double, detected_object_sptr >( score, *ix ) );
    }
  } // end foreach

  // Sort on score
  std::sort( data.begin(), data.end(), more_first< double,  detected_object_sptr >() );

  // Create new vector for return
  std::vector< detected_object_sptr > vect;

  for( auto i : data )
  {
    vect.push_back( i.second );
  }

  return std::make_shared< detected_object_set > (vect);
}

// ------------------------------------------------------------------
void
detected_object_set::
scale( double scale_factor )
{
  if( scale_factor == 1.0 )
  {
    return;
  }

  for( auto detection : m_detected_objects )
  {
    auto bbox = detection->bounding_box();
    bbox = kwiver::vital::scale( bbox, scale_factor );
    detection->set_bounding_box( bbox );
  }
}

// ------------------------------------------------------------------
void
detected_object_set::
shift( double col_shift, double row_shift )
{
  if( col_shift == 0.0 && row_shift == 0.0 )
  {
    return;
  }

  for( auto detection : m_detected_objects )
  {
    auto bbox = detection->bounding_box();
    bbox = kwiver::vital::translate( bbox,
      bounding_box_d::vector_type( col_shift, row_shift ) );
    detection->set_bounding_box( bbox );
  }
}

// ------------------------------------------------------------------
kwiver::vital::attribute_set_sptr
detected_object_set::
attributes() const
{
  return m_attrs;
}


// ------------------------------------------------------------------
void
detected_object_set::
set_attributes( attribute_set_sptr attrs )
{
  m_attrs = attrs;
}


// ----------------------------------------------------------------------------
detected_object_sptr
detected_object_set::
at( size_t pos )
{
  return m_detected_objects.at( pos );
}


// ----------------------------------------------------------------------------
const detected_object_sptr
detected_object_set::
at( size_t pos ) const
{
  return m_detected_objects.at( pos );
}


using vec_t = std::vector< detected_object_sptr >;

// ----------------------------------------------------------------------------
// Next value function for non-const iteration.
detected_object_set::iterator::next_value_func_t
detected_object_set
::get_iter_next_func()
{
  vec_t::iterator it = m_detected_objects.begin();
  return [=] () mutable ->iterator::reference {
    if( it == m_detected_objects.end() )
    {
      VITAL_THROW( stop_iteration_exception, "detected_object_set" );
    }
    return *(it++);
  };
}

// ----------------------------------------------------------------------------
// Next value function for const iteration.
detected_object_set::const_iterator::next_value_func_t
detected_object_set
::get_const_iter_next_func() const
{
  vec_t::const_iterator cit = m_detected_objects.begin();
  return [=] () mutable ->const_iterator::reference {
    if( cit == m_detected_objects.end() )
    {
      VITAL_THROW( stop_iteration_exception, "detected_object_set" );
    }
    return *(cit++);
  };
}

} } // end namespace
