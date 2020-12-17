// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
