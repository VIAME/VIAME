// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
