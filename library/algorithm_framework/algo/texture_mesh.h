// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// Interface to abstract texture mesh algorithm

#ifndef VITAL_ALGO_TEXTURE_MESH_H
#define VITAL_ALGO_TEXTURE_MESH_H

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/core_types/camera.h>
#include <viame/core_types/image_container.h>
#include <viame/core_types/mesh_container.h>
#include <viame/algorithm_framework/vital_config.h>

namespace kwiver {

namespace vital {

namespace algo {

/// Project camera frames onto a UV-unwrapped mesh to produce texture images.
class VITAL_ALGO_EXPORT texture_mesh
  : public kwiver::vital::algorithm
{
public:
  texture_mesh();
  PLUGGABLE_INTERFACE( texture_mesh );

  /// Project a single camera frame onto the mesh UV space, writing colors into
  /// \p output_image.
  ///
  /// \param mesh_container Mesh container with UV coordinates.
  /// \param output_image Pre-allocated RGBA output texture atlas.
  /// \param frame Video frame corresponding to \p camera.
  /// \param camera Camera corresponding to \p frame.
  virtual void
  texture(
    mesh_container_sptr mesh_container,
    image_container_sptr output_image,
    image_container_sptr frame,
    camera_sptr camera ) = 0;

  /// Project multiple frames onto the mesh and aggregate into one or more
  /// texture atlases. The aggregation mode is implementation-defined.
  ///
  /// \param mesh_container Mesh container with UV coordinates.
  /// \param [in,out] output_images Output texture atlas list.
  /// \param frames Video frames corresponding to \p cameras.
  /// \param cameras corresponding to \p frames.
  virtual void
  texture_list(
    mesh_container_sptr mesh_container,
    image_container_sptr_list& output_images,
    image_container_sptr_list const& frames,
    camera_sptr_list const& cameras ) = 0;

  /// Generate a float texture map encoding the 3D world-space surface position
  /// (XYZ + validity alpha) at each UV texel.
  ///
  /// \param mesh_container Mesh container with UV coordinates.
  /// \param output_image Pre-allocated 4-channel float image;
  ///   RGB channels receive world XYZ, alpha is 1 where mesh surface is
  ///   present.
  virtual void
  texture_xyz(
    mesh_container_sptr mesh_container,
    image_container_sptr output_image ) = 0;
};

typedef std::shared_ptr< texture_mesh > texture_mesh_sptr;

} // namespace algo

} // namespace vital

} // namespace kwiver

#endif
