// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface for detected_object_set
 */

#ifndef VITAL_DETECTED_OBJECT_SET_H_
#define VITAL_DETECTED_OBJECT_SET_H_

#include <vital/vital_export.h>
#include <vital/vital_config.h>
#include <vital/attribute_set.h>
#include <vital/noncopyable.h>
#include <vital/set.h>

#include <vital/types/detected_object.h>

namespace kwiver {
namespace vital {

// forward declaration of detected_object class
class detected_object_set;

// typedef for a detected_object shared pointer
using detected_object_set_sptr = std::shared_ptr< detected_object_set >;
using detected_object_set_scptr = std::shared_ptr< detected_object_set const >;

// ----------------------------------------------------------------------------
/// Set of detected objects.
///
/// This class represents a ordered set of detected objects. The detections are
/// ordered on their basic confidence value.
///
/// \par Reentrancy considerations:
///   Typical usage for a set is for a single detector thread to create a set.
///   It is possible to have an application where two threads are accessing the
///   same set concurrently.
class VITAL_EXPORT detected_object_set
  : public set< detected_object_sptr >
  , private noncopyable
{
public:

  /// Create an empty detection set.
  ///
  /// This constructor creates an empty detection set. Detections can be added
  /// with the add() method.
  detected_object_set();

  ~detected_object_set() = default;

  /// Create a new set of detected objects.
  ///
  /// This constructor creates a detection set using the supplied vector of
  /// detection objects. This can be used to create a new detection set from
  /// the output of a select() method.
  ///
  /// \param objs Vector of detected objects.
  detected_object_set( std::vector< detected_object_sptr > const& objs );

  /// Clone the detected object set (polymorphic copy constructor).
  detected_object_set_sptr clone() const;

  /// Add detection to set.
  ///
  /// This method adds a new detection to this set.
  ///
  /// \param object Detection to be added to set.
  void add( detected_object_sptr object );

  /// Add detection to set.
  ///
  /// This method adds a new detection to this set.
  ///
  /// \param detections Detection set to be added to set.
  void add( detected_object_set_sptr detections );

  /// Get number of detections in this set.
  size_t size() const override;

  /// Test if this set is empty.
  ///
  /// \return \c true if the set is empty, otherwise \c false.
  bool empty() const override;

  /// Get pointer to detected object at specified index.
  ///
  /// This method returns a reference to the element at \p pos, with bounds
  /// checking. If \p pos is not within the range of the container, a
  /// std::out_of_range exception is thrown.
  ///
  /// \param pos Position of element to return (from zero).
  ///
  /// \return Shared pointer to specified element.
  ///
  /// \throws std::out_of_range if \p pos is not within the range of objects in
  ///         the container.
  ///@{
  detected_object_sptr at( size_t pos ) override;
  const detected_object_sptr at( size_t pos ) const override;
  ///@}

  /// Select detections based on confidence value.
  ///
  /// This method returns a vector of detections ordered by confidence value,
  /// from high to low. If the optional \p threshold is specified, then all
  /// detections from the set that are less than the threshold are not in the
  /// selected set. Note that the selected set may be empty.
  ///
  /// The returned vector refers to the actual detections in the set, so if you
  /// make changes to the selected set, you are also changing the object in the
  /// set. If you want a clean set of detections, call clone() first.
  ///
  /// \param threshold
  ///   Select all detections with confidence not less than this value. If this
  ///   parameter is omitted, all detections are selected.
  ///
  /// \return List of detections.
  detected_object_set_sptr select(
    double threshold = detected_object_type::INVALID_SCORE ) const;

  /// Select detections based on class name.
  ///
  /// This method returns a vector of detections that have the specified
  /// \p class_name. These detections are ordered by descending score for the
  /// name. Note that the selected set may be empty.
  ///
  /// The returned vector refers to the actual detections in the set, so if you
  /// make changes to the selected set, you are also changing the object in the
  /// set. If you want a clean set of detections, call clone() first.
  ///
  /// \param class_name Class name of detections to be selected.
  /// \param threshold
  ///   Select all detections with confidence not less than this value. If this
  ///   parameter is omitted, all detections are selected.
  ///
  /// \return List of detections.
  detected_object_set_sptr select(
    const std::string& class_name,
    double threshold = detected_object_type::INVALID_SCORE ) const;

  /// Scale all detection locations by some scale factor.
  ///
  /// This method changes the bounding boxes within all stored detections by
  /// scaling them by the specified scale factor.
  ///
  /// \note
  ///   Detections in this set can be shared by multiple sets, so scaling the
  ///   detections in this set will also scale the detection in other sets that
  ///   share the detections. To avoid this, clone() the set before shifting.
  ///
  /// \param scale Scale factor to be applied.
  ///
  /// \deprecated
  VITAL_DEPRECATED
  void scale( double scale_factor );

  /// Shift all detection locations by some translation offset.
  ///
  /// This method shifts the bounding boxes within all stored detections by a
  /// supplied column and row shift.
  ///
  /// \note
  ///   Detections in this set can be shared by multiple sets, so shifting the
  ///   detections in this set will also shift the detection in other sets that
  ///   share the detections. To avoid this, clone() the set before shifting.
  ///
  /// \param col_shift Column (a.k.a. x, i, width) translation value.
  /// \param row_shift Row (a.k.a. y, j, height) translation value.
  ///
  /// \deprecated
  VITAL_DEPRECATED
  void shift( double col_shift, double row_shift );

  /// Get attribute set.
  ///
  /// This method returns a pointer to the attribute set that is attached to
  /// this detected object set. It is possible that the pointer is null, so
  /// check before using it.
  ///
  /// \return Pointer to attribute set or \c nullptr
  attribute_set_sptr attributes() const;

  /// Attach attribute set to this detected object set.
  ///
  /// This method attaches the specified attribute set to this detected object
  /// set.
  ///
  /// \param attrs Pointer to attribute set to attach.
  void set_attributes( attribute_set_sptr attrs );

protected:
  iterator::next_value_func_t get_iter_next_func() override;
  const_iterator::next_value_func_t get_const_iter_next_func() const override;

private:
  // List of detections ordered by confidence value.
  std::vector< detected_object_sptr > m_detected_objects;

  attribute_set_sptr m_attrs;
};

} } // end namespace

#endif
