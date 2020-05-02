/*ckwg +29
 * Copyright 2013-2018 by Kitware, Inc.
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

/**
 * \file
 * \brief core descriptor_set interface implementation
 */

#include "descriptor_set.h"
#include <vital/exceptions.h>

#include <sstream>

namespace kwiver {
namespace vital {

  // ----------------------------------------------------------------------------
descriptor_set
::descriptor_set()
  : m_logger( kwiver::vital::get_logger( "vital.descriptor_set" ) )
{
}

descriptor_set
::~descriptor_set()
{ }

// ----------------------------------------------------------------------------
kwiver::vital::logger_handle_t
descriptor_set::
logger()
{
  return m_logger;
}

// ============================================================================
// Constructor from a vector of descriptors
simple_descriptor_set
::simple_descriptor_set( const std::vector< descriptor_sptr > & descriptors )
  : data_( descriptors )
{}


simple_descriptor_set
::~simple_descriptor_set()
{ }

// ----------------------------------------------------------------------------
// Get the number of elements in this set.
size_t
simple_descriptor_set
::size() const
{
  return data_.size();
}

// ----------------------------------------------------------------------------
// Whether or not this set is empty.
bool
simple_descriptor_set
::empty() const
{
  return data_.empty();
}

// ----------------------------------------------------------------------------
// Return the descriptor at the specified index.
descriptor_sptr
simple_descriptor_set
::at( size_t index )
{
  if( index >= size() )
  {
    std::stringstream ss;
    ss << index;
    throw std::out_of_range( ss.str() );
  }
  return data_[index];
}

// ----------------------------------------------------------------------------
// Return the descriptor at the specified index.
descriptor_sptr const
simple_descriptor_set
::at( size_t index ) const
{
  if( index >= size() )
  {
    std::stringstream ss;
    ss << index;
    throw std::out_of_range( ss.str() );
  }
  return data_[index];
}

// ----------------------------------------------------------------------------
// Next value function for non-const iteration.
simple_descriptor_set::iterator::next_value_func_t
simple_descriptor_set
::get_iter_next_func()
{
  vec_t::iterator it = data_.begin();
  return [=] () mutable ->iterator::reference {
    if( it == data_.end() )
    {
      VITAL_THROW( stop_iteration_exception, "descriptor_set");
    }
    return *(it++);
  };
}

// ----------------------------------------------------------------------------
// Next value function for const iteration.
simple_descriptor_set::const_iterator::next_value_func_t
simple_descriptor_set
::get_const_iter_next_func() const
{
  vec_t::const_iterator cit = data_.begin();
  return [=] () mutable ->const_iterator::reference {
    if( cit == data_.end() )
    {
      VITAL_THROW( stop_iteration_exception, "descriptor_set" );
    }
    return *(cit++);
  };
}

} // end namespace: vital
} // end namespace: kwiver
