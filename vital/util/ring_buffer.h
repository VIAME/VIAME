// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Base interface for simple buffer classes
 */

#ifndef KWIVER_VITAL_UTIL_RING_BUFFER_H_
#define KWIVER_VITAL_UTIL_RING_BUFFER_H_

#include <vector>
#include <exception>

#include "buffer.h"

namespace kwiver {
namespace vital {


template< class Data >
class ring_buffer : public buffer< Data >
{
public:
  ring_buffer() : head_( 0 ), item_count_( 0 ) {}
  virtual ~ring_buffer() {}

  /// \brief Set the maximum capacity of the buffer.
  virtual void set_capacity( unsigned capacity )
  {
    buffer_.resize( capacity );
  }

  /// \brief Returns the maximum capacity of the buffer.
  virtual size_t capacity() const
  {
    return static_cast<unsigned>( buffer_.size() );
  }

  /// \brief Returns the number of entries in the buffer.
  virtual size_t size() const
  {
    return item_count_;
  }

  /// \brief Inserts an item into the buffer.
  ///
  /// Note: This will overwrite the oldest item in the buffer if
  /// the buffer is at capacity.
  virtual void insert( Data const& item )
  {
    ++head_;

    unsigned const N = capacity();

    if ( N <= head_ )
    {
      head_ %= N;
    }

    buffer_[head_] = item;

    if ( item_count_ < N )
    {
      ++item_count_;
    }
  }

  /// \brief The index of of the newest item in the buffer.
  virtual unsigned head() const
  {
    return head_;
  }

  /// \brief Return the item \a offset away from the most recent item.
  ///
  /// An \a offset of 0 refers to the most recent (newest) item.
  ///
  /// It is an error to ask for an offset beyond the number of items
  /// currently in the buffer.  Use has_datum_at() or size() to
  /// verify.
  virtual const Data& datum_at( unsigned offset ) const
  {
#ifdef DEBUG
    if( !has_datum_at( offset ) )
    {
      throw std::runtime_error( "No datum for specified offset" );
    }
#endif

    unsigned const N = capacity();

    int idx = static_cast<int>(head_) -
              static_cast<int>(offset) +
              static_cast<int>(N);

    if( idx < 0 )
    {
      throw std::runtime_error( "Ring buffer logic error" );
    }

    return buffer_[ static_cast<unsigned>( idx % N ) ];
  }

  /// \brief Check if there is an \a offset away from the head item.
  ///
  /// An \a offset of 0 refers to the most recent (newest) item.
  virtual bool has_datum_at( unsigned offset ) const
  {
    return ( offset < item_count_ );
  }

  /// \brief Not implemented
  virtual size_t offset_of( Data const& ) const
  {
    return static_cast< size_t >( -1 );
  }

  /// \brief Empties the buffer and returns the buffer to the initial state.
  virtual void clear()
  {
    head_ = 0;
    item_count_ = 0;
  }

protected:
  std::vector< Data > buffer_;

  unsigned head_;
  size_t item_count_;
};


} // end namespace vital
} // end namespace kwiver

#endif // KWIVER_VITAL_UTIL_RING_BUFFER_H_
