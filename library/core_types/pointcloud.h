// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Header for \link kwiver::vital::pointcloud pointcloud \endlink
/// objects

#ifndef VITAL_POINTCLOUD_H_
#define VITAL_POINTCLOUD_H_

#include <viame/core_types/vital_types_export.h>
#include <viame/algorithm_framework/vital_config.h>

#include <memory>
#include <vector>

#include "color.h"
#include "vector.h"

namespace kwiver {

namespace vital {

/// forward declaration of pointcloud class
class pointcloud;
/// typedef for a pointcloud shared pointer
typedef std::shared_ptr< pointcloud > pointcloud_sptr;

/// An abstract representation of a point cloud.
///
/// The base class pointcloud is abstract and provides a
/// double precision interface.  The templated derived class
/// can store values in either single or double precision.
class pointcloud
{
public:
  /// Destructor
  virtual
  ~pointcloud() = default;

  /// Create a clone of this pointcloud object
  virtual pointcloud_sptr clone() const = 0;

  /// Access the type info of the underlying data (double or float)
  virtual std::type_info const& data_type() const = 0;

  /// Accessor for the coordinates
  virtual std::vector< vector_3d > positions() const = 0;
  /// Accessor for the RGB colors
  virtual std::vector< rgb_color > colors() const = 0;
  /// Accessor for the intensities
  virtual std::vector< uint8_t > intensities() const = 0;

  /// Does the point cloud contain color data
  virtual bool has_colors() const = 0;
  /// Does the point cloud contain intensity data
  virtual bool has_intensities() const = 0;
};

/// A representation of a point cloud
template < typename T >
class VITAL_TYPES_EXPORT pointcloud_ :

  public pointcloud
{
public:
  /// Default Constructor
  pointcloud_();

  /// Constructor for a pointcloud
  ///
  /// \param positions positions of the pointcloud
  pointcloud_(
    std::vector< vector_< 3, T > > const& positions );

  /// Constructor for a pointcloud_ from a base class pointcloud
  explicit pointcloud_( pointcloud const& f );

  /// Create a clone of this pointcloud object
  virtual pointcloud_sptr
  clone() const
  {
    return std::make_shared< pointcloud_< T > >( *this );
  }

  /// Access statically available type of underlying data (double or float)
  static std::type_info const& static_data_type() { return typeid( T ); }

  virtual std::type_info const&
  data_type() const { return typeid( T ); }

  /// Accessor for the world coordinates using underlying data type
  std::vector< vector_< 3, T > > const&
  get_positions() const { return pos_; }

  /// Accessor for the world coordinates
  virtual std::vector< vector_3d > positions() const;

  /// Accessor for a const reference to the RGB color
  virtual std::vector< rgb_color > const&
  get_colors() const { return colors_; }

  /// Accessor for the RGB color
  virtual std::vector< rgb_color >
  colors() const { return colors_; }

  /// Does point cloud have color data
  virtual bool
  has_colors() const { return !colors_.empty(); }

  /// Accessor for a const reference to the intensities
  virtual std::vector< uint8_t > const&
  get_intensities() const { return inten_; }

  /// Accessor for the intensities
  virtual std::vector< uint8_t >
  intensities() const { return inten_; }

  /// Does point cloud have intensity data
  virtual bool
  has_intensities() const { return !inten_.empty(); }

  /// Set the point cloud positions
  void set_positions( std::vector< vector_< 3, T > > const& pos );

  /// Set the RGB colors of the point cloud
  void set_color( std::vector< rgb_color > const& colors );

  /// Set the intensities of the point cloud
  void set_intensities( std::vector< uint8_t > const& inten );

protected:
  /// A vector representing the 3D position of the pointcloud
  std::vector< vector_< 3, T > > pos_;
  /// The RGB color associated with the pointcloud
  std::vector< rgb_color > colors_;
  /// The intensities associated with the pointcloud
  std::vector< uint8_t > inten_;
};

/// A double precision pointcloud
typedef pointcloud_< double > pointcloud_d;
/// A single precision pointcloud
typedef pointcloud_< float > pointcloud_f;

} // namespace vital

}   // end namespace vital

#endif // VITAL_POINTCLOUD_H_
