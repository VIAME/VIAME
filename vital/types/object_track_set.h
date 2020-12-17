// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header file for \link kwiver::vital::object_track_set object_track_set
 *        \endlink and a concrete \link kwiver::vital::simple_object_track_set
 *        simple_object_track_set \endlink
 */

#ifndef VITAL_OBJECT_TRACK_SET_H_
#define VITAL_OBJECT_TRACK_SET_H_

#include <vital/types/detected_object.h>
#include <vital/types/point.h>
#include <vital/types/timestamp.h>
#include <vital/types/track_set.h>

#include <vital/vital_export.h>
#include <vital/vital_config.h>
#include <vital/vital_types.h>

#include <vital/range/transform.h>

#include <vector>
#include <memory>

namespace kwiver {
namespace vital {

// ============================================================================
/// A derived track_state for object tracks
class VITAL_EXPORT object_track_state : public track_state
{
public:
  object_track_state() = default;

  //@{
  /// Default constructor
  object_track_state( frame_id_t frame,
                      time_usec_t time,
                      detected_object_sptr const& d = nullptr )
    : track_state( frame )
    , detection_( d )
    , time_( time )
  {}

  object_track_state( frame_id_t frame,
                      time_usec_t time,
                      detected_object_sptr&& d )
    : track_state( frame )
    , detection_( std::move( d ) )
    , time_( time )
  {}
  //@}

  //@{
  /// Alternative constructor
  object_track_state( timestamp const& ts,
                      detected_object_sptr const& d = nullptr )
    : track_state( ts.get_frame() )
    , detection_( d )
    , time_( ts.get_time_usec() )
  {}

  object_track_state( timestamp const& ts,
                      detected_object_sptr&& d )
    : track_state( ts.get_frame() )
    , detection_( std::move( d ) )
    , time_( ts.get_time_usec() )
  {}
  //@}

  /// Copy constructor
  object_track_state( object_track_state const& other ) = default;

  /// Move constructor
  object_track_state( object_track_state&& other ) = default;

  /// Clone the track state (polymorphic copy constructor)
  track_state_sptr clone( clone_type ct = clone_type::DEEP ) const override;

  void set_time( time_usec_t time )
  {
    time_ = time;
  }

  time_usec_t time() const
  {
    return time_;
  }

  void set_detection( detected_object_sptr const& d )
  {
    detection_ = d;
  }

  detected_object_sptr detection()
  {
    return detection_;
  }

  detected_object_scptr detection() const
  {
    return detection_;
  }

  void set_image_point( point_2d const& p )
  {
    image_point_ = p;
  }

  point_2d image_point() const
  {
    return image_point_;
  }

  void set_track_point( point_3d const& p )
  {
    track_point_ = p;
  }

  point_3d track_point() const
  {
    return track_point_;
  }

  static std::shared_ptr< object_track_state > downcast(
    track_state_sptr const& sp )
  {
    return std::dynamic_pointer_cast< object_track_state >( sp );
  }

  static constexpr auto downcast_transform = range::transform( downcast );

private:
  detected_object_sptr detection_;
  point_2d image_point_;
  point_3d track_point_;
  time_usec_t time_ = 0;
};

// ============================================================================
/// A collection of object tracks
class VITAL_EXPORT object_track_set : public track_set
{
public:
  /// Default Constructor
  /**
   * \note implementation defaults to simple_track_set_implementation
   */
  object_track_set();

  /// Constructor specifying the implementation
  object_track_set( std::unique_ptr< track_set_implementation > impl );

  /// Constructor from a vector of tracks
  /**
   * \note implementation defaults to simple_track_set_implementation
   */
  object_track_set( std::vector< track_sptr > const& tracks );

  /// Destructor
  virtual ~object_track_set() = default;
};

/// Shared pointer for object_track_set type
typedef std::shared_ptr< object_track_set > object_track_set_sptr;

/// Helper to iterate over the states of a track as object track states
/**
 * This object is an instance of a range transform adapter that can be applied
 * to a track_sptr in order to directly iterate over the underlying
 * object_track_state instances.
 *
 * \par Example:
 * \code
 * namespace kv = kwiver::vital;
 * namespace r = kwiver::vital::range;
 *
 * kv::track_sptr ot = get_the_object_track();
 * for ( auto s : ot | kv::as_object_track )
 *   std::cout << s->time() << std::endl;
 * \endcode
 *
 * \sa kwiver::vital::range::transform_view
 */
static constexpr auto as_object_track = object_track_state::downcast_transform;

} } // end namespace vital

#endif // VITAL_OBJECT_TRACK_SET_H_
