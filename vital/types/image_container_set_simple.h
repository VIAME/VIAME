// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface for simple implementation of image_container_set
 */

#ifndef VITAL_IMAGE_CONTAINER_SET_SIMPLE_H_
#define VITAL_IMAGE_CONTAINER_SET_SIMPLE_H_

#include <vital/types/image_container_set.h>

namespace kwiver {
namespace vital {

/// A concrete image container set that simply wraps a vector of images.
class VITAL_EXPORT simple_image_container_set
  : public image_container_set
{
public:
  /// Default Constructor
  simple_image_container_set();

  /// Constructor from a vector of images
  explicit simple_image_container_set( std::vector< image_container_sptr > const& images );

  /// Return the number of items
  size_t size() const override;
  bool empty() const override;
  image_container_sptr at( size_t index ) override;
  image_container_sptr const at( size_t index ) const override;

protected:
  using vec_t = std::vector< image_container_sptr >;

  /// The vector of images
  vec_t data_;

  /// Implement next function for non-const iterator.
  iterator::next_value_func_t get_iter_next_func();

  /// Implement next function for const iterator.
  const_iterator::next_value_func_t get_const_iter_next_func() const;
};

} } // end namespaces

#endif // VITAL_IMAGE_CONTAINER_SET_SIMPLE_H_
