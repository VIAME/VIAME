// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Interface for detected_object class

#ifndef VITAL_DETECTED_OBJECT_H_
#define VITAL_DETECTED_OBJECT_H_

#include <viame/core_types/vital_types_export.h>
#include <viame/algorithm_framework/vital_config.h>

#include <viame/core_types/attribute_set.h>

#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include <viame/core_types/bounding_box.h>
#include <viame/core_types/descriptor.h>
#include <viame/core_types/detected_object_type.h>
#include <viame/core_types/geo_point.h>
#include <viame/core_types/image_container.h>
#include <viame/core_types/point.h>
#include <viame/core_types/vector.h>

#include <viame/algorithm_framework/io/eigen_io.h>
namespace kwiver {

namespace vital {

// forward declaration of detected_object class
class detected_object;

// typedef for a detected_object shared pointer
using detected_object_sptr = std::shared_ptr< detected_object >;
using detected_object_scptr = std::shared_ptr< detected_object const >;

// ----------------------------------------------------------------------------
/// @brief Detected object class.
///
/// This class represents a detected object in image space.
///
/// There is one object of this type for each detected object. These
/// objects are defined by a bounding box in the image space. Each
/// object has an optional classification object attached.
class VITAL_TYPES_EXPORT detected_object
{
public:
  using vector_t = std::vector< detected_object_sptr >;
  using descriptor_t = descriptor_dynamic< double >;
  using descriptor_scptr = std::shared_ptr< descriptor_t const >;
  using notes_t = std::vector< std::string >;
  using keypoints_t = std::map< std::string, vital::point_2d >;

  /// @brief Create default detected object.
  ///
  /// @param confidence Detectors confidence in this detection.
  /// @param classifications Optional object classification.
  detected_object(
    double confidence = 1.0,
    detected_object_type_sptr classifications = nullptr );

  /// @brief Create detected object with bounding box and other attributes.
  ///
  /// @param bbox Bounding box surrounding detected object, in image
  ///             coordinates.
  /// @param confidence Detectors confidence in this detection.
  /// @param classifications Optional object classification.
  detected_object(
    bounding_box_d const& bbox,
    double confidence = 1.0,
    detected_object_type_sptr classifications = nullptr );

  /// @brief Create detected object with a geo_point and other attributes.
  ///
  /// @param geo_pt Geographic location of the detection, in world coordinates.
  /// @param confidence Detectors confidence in this detection.
  /// @param classifications Optional object classification.
  detected_object(
    kwiver::vital::geo_point const& geo_pt,
    double confidence = 1.0,
    detected_object_type_sptr classifications = nullptr );

  virtual ~detected_object() = default;

  /// @brief Copy constructor.
  ///
  /// The attribute mutex is not copyable, so copy operations are defined
  /// explicitly.  Each object retains its own mutex while the remaining
  /// members are copied.
  detected_object( detected_object const& other );

  /// @brief Copy assignment operator.
  detected_object& operator=( detected_object const& other );

  /// @brief Create a deep copy of this object.
  ///
  /// @return Managed copy of this object.
  detected_object_sptr clone() const;

  /// @brief Get bounding box from this detection.
  ///
  /// The bounding box for this detection is returned. This box is in
  /// image coordinates. A default constructed (invalid) bounding box
  /// is returned if no box has been supplied for this detection.
  ///
  /// @return A copy of the bounding box.
  bounding_box_d bounding_box() const;

  /// @brief Set new bounding box for this detection.
  ///
  /// The supplied bounding box replaces the box for this detection.
  ///
  /// @param bbox Bounding box for this detection.
  void set_bounding_box( bounding_box_d const& bbox );

  /// @brief Get geo_point from this detection.
  ///
  /// The geo_point for this detection is returned.
  /// A default constructed (invalid) geo_point is returned
  /// if no point has been supplied for this detection.
  ///
  /// @return A copy of the geo_point.
  kwiver::vital::geo_point geo_point() const;

  /// @brief Set new geo_point for this detection.
  ///
  /// The supplied geo_point replaces the point for this detection.
  ///
  /// @param gp geo_point for this detection.
  void set_geo_point( kwiver::vital::geo_point const& gp );

  /// @brief Get confidence for this detection.
  ///
  /// This method returns the current confidence value for this detection.
  /// Confidence values are in the range of 0.0 - 1.0.
  ///
  /// @return Confidence value for this detection.
  double confidence() const;

  /// @brief Set new confidence value for detection.
  ///
  /// This method sets a new confidence value for this detection.
  /// Confidence values are in the range of [0.0 - 1.0].
  ///
  /// @param d New confidence value for this detection.
  void set_confidence( double d );

  /// @brief Get detection index.
  ///
  /// This method returns the index for this detection.
  ///
  /// The detection index is a general purpose field that the
  /// application can use to individually identify a detection. In some
  /// cases, this field can be used to correlate the detection of an
  /// object over multiple frames.
  ///
  /// @return Detection index fof this detections.
  uint64_t index() const;

  /// @brief Set detection index.
  ///
  /// This method sets tne index value for this detection.
  ///
  /// The detection index is a general purpose field that the
  /// application can use to individually identify a detection. In some
  /// cases, this field can be used to correlate the detection of an
  /// object over multiple frames.
  ///
  /// @param idx Detection index.
  void set_index( uint64_t idx );

  /// @brief Get detector name.
  ///
  /// This method returns the name of the detector that created this
  /// element. An empty string is returned if the detector name is not
  /// set.
  ///
  /// @return Name of the detector.
  std::string detector_name() const;

  /// @brief Set detector name.
  ///
  /// This method sets the name of the detector for this detection.
  ///
  /// @param name Detector name.
  void set_detector_name( std::string const& name );

  /// @brief Get pointer to optional classifications object.
  ///
  /// This method returns the pointer to the classification object if
  /// there is one. If there is no classification object the pointer is
  /// NULL.
  ///
  /// @return Pointer to classification object or NULL.
  detected_object_type_sptr type() const;

  /// @brief Set new classifications for this detection.
  ///
  /// This method supplies a new set of class_names and scores for this
  /// detection.
  ///
  /// @param c New classification for this detection
  void set_type( detected_object_type_sptr c );

  /// @brief Get detection mask image.
  ///
  /// This method returns the mask image associated with this detection.
  ///
  /// @return Pointer to the mask image.
  image_container_scptr mask() const;

  /// @brief Set mask image for this detection.
  ///
  /// This method supplies a new mask image for this detection.
  ///
  /// @param m Mask image
  void set_mask( image_container_scptr m );

  /// @brief Get descriptor vector.
  ///
  /// This method returns an optional descriptor vector that was used
  /// to create this detection. This is only set for certain object
  /// detectors.
  ///
  /// @return Pointer to the descriptor vector.
  descriptor_scptr descriptor() const;

  /// @brief Set descriptor for this detection.
  ///
  /// This method sets a descriptor vector that was used to create this
  /// detection. This is only set for certain object detectors.
  ///
  /// @param d Descriptor vector
  void set_descriptor( descriptor_scptr d );

  /// @brief Get vector of notes for this detection
  ///
  /// This method returns a list of notes (arbitrary strings) associated
  /// with this detection. Notes are useful in user interfaces for making
  /// any observations about this detection which don't fit into types.
  ///
  /// @return A vector of notes.
  notes_t notes() const;

  /// @brief Add a note for this detection.
  ///
  /// Notes are useful in user interfaces for making any observations about
  /// this detection which don't fit into types.
  ///
  /// @param note String to add as a note
  void add_note( std::string const& note );

  /// @brief Reset notes for this detection
  ///
  /// Remove any notes stored within this detection.
  void clear_notes();

  /// @brief Returns a list of keypoints associated with this detection
  ///
  /// This method returns a map of keypoints associated with this detection,
  /// which can be of arbitrary length.
  ///
  /// @return A map of keypoints and their identifiers.
  keypoints_t keypoints() const;

  /// @brief Add a note for this detection
  ///
  /// Notes are useful in user interfaces for making any observations about
  /// this detection which don't fit into types. If a keypoint of the given
  /// name already exists, it will be over-written.
  ///
  /// @param id String id of the keypoint
  /// @param p The location of the keypoint
  void add_keypoint( std::string const& id, vital::point_2d const& p );

  /// @brief Reset keypoints for this detection
  ///
  /// Removes any keypoints stored within this detection.
  void clear_keypoints();

  /// @brief Get the polygon outline for this detection.
  ///
  /// This method returns the polygon outline associated with this detection.
  /// The polygon is represented as a vector of 2D points.
  ///
  /// @return Vector of polygon vertices.
  std::vector< vector_2d > polygon() const;

  /// @brief Set the polygon outline for this detection.
  ///
  /// This method sets the polygon outline for this detection.
  ///
  /// @param poly Vector of polygon vertices.
  void set_polygon( std::vector< vector_2d > const& poly );

  /// @brief Set the polygon from a flattened vector.
  ///
  /// This method sets the polygon from a flattened vector of coordinates
  /// where values are x1, y1, x2, y2, etc.
  ///
  /// @param coords Flattened vector of coordinates.
  void set_flattened_polygon( std::vector< double > const& coords );

  /// @brief Get the flattened polygon coordinates.
  ///
  /// This method returns the polygon as a flattened vector of coordinates
  /// where values are x1, y1, x2, y2, etc.
  ///
  /// @return Flattened vector of coordinates.
  std::vector< double > get_flattened_polygon() const;

  /// @brief Get attribute set.
  ///
  /// This method returns a pointer to the attribute set that is attached to
  /// this detected object. It is possible that the pointer is null, so
  /// check before using it.
  ///
  /// @return Pointer to attribute set or \c nullptr
  attribute_set_sptr attributes() const;

  /// @brief Attach attribute set to this detected object.
  ///
  /// This method attaches the specified attribute set to this detected object.
  ///
  /// @param attrs Pointer to attribute set to attach.
  void set_attributes( attribute_set_sptr attrs );

  /// @brief Set a single attribute value.
  ///
  /// This method sets a single attribute value in the attribute set. If no
  /// attribute set exists for this detection, one is created automatically.
  /// This method is thread-safe.
  ///
  /// @param key The attribute name/key.
  /// @param value The attribute value.
  ///
  /// @tparam T The type of the attribute value.
  template < typename T >
  void
  set_attribute( std::string const& key, T const& value )
  {
    std::lock_guard< std::mutex > lock( m_attrs_mutex );
    if( !m_attrs )
    {
      m_attrs = std::make_shared< attribute_set >();
    }
    m_attrs->add( key, value );
  }

  /// @brief Check if an attribute exists.
  ///
  /// This method checks if the named attribute exists in the attribute set.
  /// Returns false if no attribute set exists or if the attribute is not found.
  ///
  /// @param key The attribute name/key.
  ///
  /// @return \b true if the attribute exists, \b false otherwise.
  bool has_attribute( std::string const& key ) const;

  /// @brief Get a single attribute value.
  ///
  /// This method retrieves a single attribute value from the attribute set.
  ///
  /// @param key The attribute name/key.
  ///
  /// @tparam T The expected type of the attribute value.
  ///
  /// @return The attribute value.
  ///
  /// @throws attribute_set_exception if no attribute set exists or if the
  ///         named attribute is not in the set.
  /// @throws kwiver::vital::bad_any_cast if actual type does not match
  ///         requested type.
  template < typename T >
  T
  get_attribute( std::string const& key ) const
  {
    std::lock_guard< std::mutex > lock( m_attrs_mutex );
    if( !m_attrs )
    {
      throw attribute_set_exception(
        "No attribute set exists for this detection" );
    }
    return m_attrs->get< T >( key );
  }

private:
  kwiver::vital::geo_point m_geo_point;
  bounding_box_d m_bounding_box;
  double m_confidence;
  image_container_scptr m_mask_image;
  descriptor_scptr m_descriptor;

  // The detection type is an optional list of possible object types.
  detected_object_type_sptr m_type;

  uint64_t m_index = 0; ///< index for this object
  std::string m_detector_name;

  std::vector< std::string > m_notes;
  std::map< std::string, vital::point_2d > m_keypoints;
  std::vector< vector_2d > m_polygon;

  attribute_set_sptr m_attrs;
  mutable std::mutex m_attrs_mutex; ///< mutex for thread-safe attribute access
};

} // namespace vital

} // namespace kwiver

#endif
