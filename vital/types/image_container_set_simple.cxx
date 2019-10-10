/*ckwg +29
 * Copyright 2018, 2019 by Kitware, Inc.
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

#include "image_container_set_simple.h"

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------------------
// Default Constructor
simple_image_container_set::
simple_image_container_set()
{ }

// Constructor from a vector of images
simple_image_container_set::
simple_image_container_set( std::vector< image_container_sptr > const& images )
  : data_( images )
{ }

// ----------------------------------------------------------------------------
// Return the number of items
size_t
simple_image_container_set::
size() const
{
  return data_.size();
}

// ----------------------------------------------------------------------------
bool
simple_image_container_set::
empty() const
{
  return data_.empty();
}


// ----------------------------------------------------------------------------
image_container_sptr
simple_image_container_set::
at( size_t index )
{
  return data_.at(index);
}


// ----------------------------------------------------------------------------
image_container_sptr const
simple_image_container_set::
at( size_t index ) const
{
  return data_.at(index);
}

// ----------------------------------------------------------------------------
// Implement next function for non-const iterator.
simple_image_container_set::iterator::next_value_func_t
simple_image_container_set::
get_iter_next_func()
{
  vec_t::iterator v_it = data_.begin();
  // Lambda notes:
  // - [=] capture by copy
  // - mutable: modify the parameters captured by copy
  return [=] () mutable -> iterator::reference {
    if ( v_it == data_.end() )
    {
      VITAL_THROW( stop_iteration_exception, "simple_image_container_set" );
    }
    return *( v_it++ );
  };
}

// ----------------------------------------------------------------------------
// Implement next function for const iterator.
simple_image_container_set::const_iterator::next_value_func_t
simple_image_container_set::
get_const_iter_next_func() const
{
  vec_t::const_iterator v_cit = data_.begin();
  // Lambda notes:
  // - [=] capture by copy
  // - mutable: modify the parameters captured by copy
  return [=] () mutable -> const_iterator::reference {
    if ( v_cit == data_.end() )
    {
      VITAL_THROW( stop_iteration_exception, "simple_image_container_set" );
    }
    return *( v_cit++ );
  };
}


} } // end namespace
