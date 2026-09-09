// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Header for \link kwiver::vital::camera_rig camera_rig \endlink

#ifndef VITAL_CAMERA_RIG_H_
#define VITAL_CAMERA_RIG_H_

#include "camera.h"
#include <viame/core_types/vital_types_export.h>

#include <unordered_map>

namespace kwiver {

namespace vital {

/// forward declaration of camera_rig class
class camera_rig;
/// camera_rig shared pointer
using camera_rig_sptr = std::shared_ptr< camera_rig >;
/// collection of cameras keyed by string tags
using camera_collection = std::unordered_map< std::string, camera_sptr >;

// ----------------------------------------------------------------------------
/// A representation of camera rig
///
/// The base class for camera rigs
class camera_rig
{
public:
  camera_rig() = default;
  virtual ~camera_rig() = default;

  /// Access cameras of the rig.
  /// \return camera collection
  virtual camera_collection const&
  cameras() const { return cameras_; }

  /// Add a camera.
  /// \return inserted camera
  virtual camera_sptr
  add( std::string const& tag, camera_sptr camera )
  {
    return cameras_[ tag ] = camera;
  }

  /// Remove a particular camera.
  /// \return removed camera
  virtual camera_sptr
  remove( std::string const& tag )
  {
    auto it = cameras_.find( tag );
    if( it == cameras_.end() ) { return nullptr; }

    auto res = it->second;
    cameras_.erase( it );
    return res;
  }

  /// Access a particular camera.
  /// \return camera at tag, or nullptr, if none found
  virtual camera_sptr
  camera( std::string const& tag )
  {
    auto it = cameras_.find( tag );
    return it == cameras_.end() ? nullptr : it->second;
  }

  /// Access a particular camera.
  /// \return camera at tag, or nullptr, if none found
  virtual camera_sptr const
  camera( std::string const& tag ) const
  {
    auto it = cameras_.find( tag );
    return it == cameras_.end() ? nullptr : it->second;
  }

  size_t
  size() const { return cameras_.size(); }
  bool
  empty() const { return !size(); }

protected:
  camera_collection cameras_;
};

// ----------------------------------------------------------------------------
/// A representation of camera stereo rig
///
/// A camera stereo rig class
class VITAL_TYPES_EXPORT camera_rig_stereo : public camera_rig
{
public:
  /// Construct a stereo rig using left and right cameras.
  camera_rig_stereo( camera_sptr left, camera_sptr right );
  virtual ~camera_rig_stereo();
  using camera_rig::cameras;
  using camera_rig::camera;
  camera_sptr left() { return camera( "left" ); }
  camera_sptr right() { return camera( "right" ); }
  camera_sptr const
  left()const { return camera( "left" ); }
  camera_sptr const
  right()const { return camera( "right" ); }
};

using camera_rig_stereo_sptr = std::shared_ptr< camera_rig_stereo >;

} // vital

} // kwiver

#endif // VITAL_CAMERA_RIG_H_
