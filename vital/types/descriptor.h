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
 * \brief core descriptor interface and template implementations
 */

#ifndef VITAL_DESCRIPTOR_H_
#define VITAL_DESCRIPTOR_H_

#include <vital/vital_export.h>
#include <vital/vital_config.h>
#include <vital/vital_types.h>
#include <vital/exceptions.h>

#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include <cstring>

namespace kwiver {
namespace vital {

/// Shared pointer for base descriptor type
class descriptor;
typedef std::shared_ptr< descriptor > descriptor_sptr;

// ------------------------------------------------------------------
/// A representation of a feature descriptor used in matching.
class descriptor
{
public:
  /// Destructor
  virtual ~descriptor() = default;

  virtual descriptor_sptr clone() const = 0;

  /// Access the type info of the underlying data (double or float)
  virtual std::type_info const& data_type() const = 0;

  /// The number of elements of the underlying type
  virtual std::size_t size() const = 0;

  /// The number of bytes used to represent the data
  virtual std::size_t num_bytes() const = 0;

  /// Return the descriptor as pointer to bytes
  /**
   * Subclasses should ensure this always works by storing the data
   * as a continuous byte array.
   * Note that as_bytes returns a pointer to the underlying data while
   * as_double returns a vector of doubles which will be copied from
   * the underlying data if possible.  As_bytes is written this way
   * for speed (no copying) at the cost of being restrictive on sub-classes
   * in terms of the way they lay out their descriptors in memory.
   */
  virtual const byte* as_bytes() const = 0;

  /// Return the descriptor as a vector of doubles
  /**
   * Return an empty vector if this makes no sense
   * for the underlying type.
   */
  virtual std::vector< double > as_double() const = 0;

  /// Equality operator
  bool operator==( descriptor const& other ) const
  {
    if( this->data_type() != other.data_type() ||
        this->size() != other.size() )
    {
      return false;
    }
    auto b1 = this->as_bytes();
    auto b2 = other.as_bytes();

    return std::equal(b1, b1 + this->num_bytes(), b2);
  }

  /// Inequality operator
  bool operator!=( descriptor const& other ) const
  {
    return ! operator==(other);
  }

  /// Returns the node_id for the descriptor.
  /**
   * The node_id is generally the vocabulary tree leaf index computed when
   * the descriptor is quantized in the tree.  Two features with the same
   * node_id are expected to have similar visual appearance.
  */
  virtual unsigned int node_id() const { return 0; }

  /// Sets the node_id for the descriptor.
  /**
   * By default this returns false because this base class has nowhere
   * to store the node_id.  Derived classes that do store the node_id
   * should return true if it successfully stored.
  */
  virtual bool set_node_id(unsigned int node_id) { return false; }

};


// ------------------------------------------------------------------
/// Abstract base class of a descriptor containing an array of type T
template < typename T >
class descriptor_array_of :
  public descriptor
{
public:
  /// Access the type info of the underlying data (double or float)
  virtual std::type_info const& data_type() const { return typeid( T ); }

  /// The number of bytes used to represent the data
  std::size_t num_bytes() const { return this->size() * sizeof( T ); }


  /// Return the descriptor as a vector of doubles
  std::vector< double > as_double() const
  {
    const std::size_t length = this->size();
    std::vector< double > double_data( length );

    for ( std::size_t i = 0; i < length; ++i )
    {
      double_data[i] = static_cast< double > ( this->raw_data()[i] );
    }
    return double_data;
  }

  virtual const byte* as_bytes() const
  {
    return reinterpret_cast<const byte *>(raw_data());
  }

  /// Return an pointer to the raw data array
  virtual T* raw_data() = 0;

  /// Return an pointer to the raw data array
  virtual const T* raw_data() const = 0;

  // Iterator interface
  T const* begin() const { return this->raw_data(); }
  T const* end() const { return this->raw_data() + this->size(); }


  /// Equality operator
  bool operator==( descriptor_array_of<T> const& other ) const
  {
    if( this->size() != other.size() )
    {
      return false;
    }
    return std::equal(this->raw_data(), this->raw_data() + this->size(),
                      other.raw_data());
  }

  /// Inequality operator
  bool operator!=( descriptor_array_of<T> const& other ) const
  {
    return ! operator==(other);
  }

};


// ------------------------------------------------------------------
/// A representation of a descriptor of fixed type and size
template < typename T, unsigned N >
class descriptor_fixed :
  public descriptor_array_of< T >
{
public:
  /// Default Constructor
  descriptor_fixed< T, N > ( ):
    node_id_(std::numeric_limits<unsigned int>::max()) { }

  /// The number of elements of the underlying type
  std::size_t size() const { return N; }

  /// Return an pointer to the raw data array
  T* raw_data() { return data_; }

  /// Return an pointer to the raw data array
  const T* raw_data() const { return data_; }

  virtual descriptor_sptr clone() const
  {
    auto new_desc = std::make_shared<descriptor_fixed<T, N>>();
    memcpy(&new_desc->data_, &data_, N*sizeof(T));
    return new_desc;
  }

  virtual unsigned int node_id() const { return node_id_; }

  virtual bool set_node_id(unsigned int node_id)
  {
    node_id_ = node_id;
    return true;
  }

protected:
  /// data array
  T data_[N];
  /// node id
  unsigned int node_id_;
};


// ------------------------------------------------------------------
/// A representation of a descriptor of fixed type and variable size
template < typename T >
class descriptor_dynamic :
  public descriptor_array_of< T >
{
public:
  /// Constructor
  descriptor_dynamic< T > (size_t len)
  : data_( new T[len] ),
  length_( len ),
  node_id_(std::numeric_limits<unsigned int>::max()) { }

  descriptor_dynamic< T > (size_t len, T* dat)
  : length_( len )
  {
    data_ = new T[len];
    memmove( data_, dat, len*sizeof(T) );
  }

  /// Destructor
  virtual ~descriptor_dynamic< T > ( ) { delete [] data_; }

  /// The number of elements of the underlying type
  std::size_t size() const { return length_; }

  /// Return an pointer to the raw data array
  T* raw_data() { return data_; }

  /// Return an pointer to the raw data array
  const T* raw_data() const { return data_; }

  virtual  descriptor_sptr clone() const
  {
    auto ptr = std::make_shared<descriptor_dynamic<T>>(length_);
    memcpy(ptr->data_, data_, length_ * sizeof(T));
    return ptr;
  }

  virtual unsigned int node_id() const { return node_id_; }

  virtual bool set_node_id(unsigned int node_id)
  {
    node_id_ = node_id;
    return true;
  }

protected:
  /// data array
  T* data_;
  /// length of data array
  size_t length_;
  /// node id
  unsigned int node_id_;
};

/// return the hamming_distance between two descriptors
VITAL_EXPORT
int hamming_distance(vital::descriptor_sptr d1, vital::descriptor_sptr d2);

// ------------------------------------------------------------------
/// output stream operator for a feature
VITAL_EXPORT std::ostream& operator<<( std::ostream& s, const descriptor& d );

/// input stream operator for a feature
VITAL_EXPORT std::istream& operator>>( std::istream& s, descriptor& d );

} } // end namespace vital

#endif // VITAL_DESCRIPTOR_H_
