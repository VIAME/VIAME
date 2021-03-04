// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief core descriptor_set interface
 */

#ifndef VITAL_DESCRIPTOR_SET_H_
#define VITAL_DESCRIPTOR_SET_H_

#include "descriptor.h"
#include <vital/set.h>
#include <vital/logger/logger.h>
#include <vital/vital_export.h>

namespace kwiver {
namespace vital {

/// An abstract ordered collection of feature descriptors.
/**
 * The base class descriptor_set is abstract and provides an interface
 * for returning a vector of descriptors.  There is a simple derived class
 * that stores the data as a vector of descriptors and returns it.  Other
 * derived classes can store the data in other formats and convert on demand.
 */
class VITAL_EXPORT descriptor_set
  : public set< descriptor_sptr >
{
public:
  /// Destructor
  virtual ~descriptor_set();

  /// Return a vector of descriptor shared pointers
  /**
   * This is not the best way to get a list of descriptors as it
   * breaks the data abstraction and would be difficult to implement
   * if the backing store was a data base.
   */
  virtual std::vector< descriptor_sptr > descriptors() const = 0;

protected:
  descriptor_set();

  kwiver::vital::logger_handle_t logger();

private:
  kwiver::vital::logger_handle_t m_logger;
};

/// Shared pointer for base descriptor_set type
typedef std::shared_ptr< descriptor_set > descriptor_set_sptr;

// ============================================================================
/// A concrete descriptor set that simply wraps a vector of descriptors.
class VITAL_EXPORT simple_descriptor_set :
  public descriptor_set
{
public:
  /// Default Constructor
  simple_descriptor_set() = default;
  ~simple_descriptor_set();

  /// Constructor from a vector of descriptors
  explicit simple_descriptor_set( const std::vector< descriptor_sptr >& descriptors );

  /**
   * Get the number of elements in this set.
   *
   * @returns Number of elements in this set.
   */
  size_t size() const override;

  /**
   * Whether or not this set is empty.
   *
   * @return True if this set is empty or false otherwise.
   */
  bool empty() const override;

  //@{
  /**
   * Return the descriptor at the specified index.
   * @param index 0-based index to access.
   * @return The descriptor shared pointer at the specified index.
   * @throws std::out_of_range If position is now within the range of objects
   *                           in container.
   */
  descriptor_sptr at( size_t index ) override;
  descriptor_sptr const at( size_t index ) const override;
  //@}

  /// Return a vector of descriptor shared pointers
  virtual std::vector< descriptor_sptr > descriptors() const { return data_; }

protected:
  using vec_t = std::vector< descriptor_sptr >;

  /// The vector of descriptors
  vec_t data_;

  /**
   * Next value function for non-const iteration.
   */
  iterator::next_value_func_t get_iter_next_func();

  /**
   * Next value function for const iteration.
   */
  const_iterator::next_value_func_t get_const_iter_next_func() const;
};

} } // end namespace vital

#endif // VITAL_DESCRIPTOR_SET_H_
