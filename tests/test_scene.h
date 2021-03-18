// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 *
 * \brief Various functions for creating a simple SBA test scene
 *
 * These functions are based on VITAL core and shared by various tests
 */

#ifndef VITAL_TEST_TEST_SCENE_H_
#define VITAL_TEST_TEST_SCENE_H_

#include "test_random_point.h"

#include <vital/math_constants.h>
#include <vital/types/camera_map.h>
#include <vital/types/camera_perspective.h>
#include <vital/types/feature_track_set.h>
#include <vital/types/landmark_map.h>
#include <vital/types/mesh.h>

namespace kwiver {
namespace testing {

// construct a map of landmarks at the corners of a cube centered at c
// with a side length of s
kwiver::vital::landmark_map_sptr
cube_corners( double s, const kwiver::vital::vector_3d& c = kwiver::vital::vector_3d(0, 0, 0) )
{
  using namespace kwiver::vital;

  // create corners of a cube
  landmark_map::map_landmark_t landmarks;
  s /= 2.0;
  landmarks[0] = landmark_sptr( new landmark_d( c + vector_3d( -s, -s, -s ) ) );
  landmarks[1] = landmark_sptr( new landmark_d( c + vector_3d( -s, -s,  s ) ) );
  landmarks[2] = landmark_sptr( new landmark_d( c + vector_3d( -s,  s, -s ) ) );
  landmarks[3] = landmark_sptr( new landmark_d( c + vector_3d( -s,  s,  s ) ) );
  landmarks[4] = landmark_sptr( new landmark_d( c + vector_3d( s, -s, -s ) ) );
  landmarks[5] = landmark_sptr( new landmark_d( c + vector_3d( s, -s,  s ) ) );
  landmarks[6] = landmark_sptr( new landmark_d( c + vector_3d( s,  s, -s ) ) );
  landmarks[7] = landmark_sptr( new landmark_d( c + vector_3d( s,  s,  s ) ) );

  return landmark_map_sptr( new simple_landmark_map( landmarks ) );
}

// construct a cube mesh centered at c with a side length of s
kwiver::vital::mesh_sptr
cube_mesh(double s, const kwiver::vital::vector_3d& c = { 0.0, 0.0, 0.0 })
{
  using namespace kwiver::vital;
  s /= 2.0;
  auto verts = new mesh_vertex_array<3> { {-s, -s, -s},
                                          {-s, -s,  s},
                                          {-s,  s, -s},
                                          {-s,  s,  s},
                                          { s, -s, -s},
                                          { s, -s,  s},
                                          { s,  s, -s},
                                          { s,  s,  s} };
  for (auto & vert : *verts)
  {
    vert += c;
  }
  auto faces = new mesh_regular_face_array<4> { {0, 1, 3, 2},
                                                {4, 6, 7, 5},
                                                {5, 7, 3, 1},
                                                {6, 4, 0, 2},
                                                {7, 6, 2, 3},
                                                {1, 0, 4, 5} };
  return std::make_shared<mesh>(std::unique_ptr<mesh_vertex_array_base>(verts),
                                std::unique_ptr<mesh_face_array_base>(faces));
}

// construct a square mesh in XY centered at c with a side length of s
kwiver::vital::mesh_sptr
grid_mesh(unsigned width, unsigned height, double scale = 1.0,
          const kwiver::vital::vector_3d& origin = { 0.0, 0.0, 0.0 })
{
  using namespace kwiver::vital;
  auto verts = new mesh_vertex_array<3>;
  auto faces = new mesh_regular_face_array<3>;
  unsigned index = 0;
  for (unsigned h = 0; h <= height; ++h)
  {
    for (unsigned w = 0; w <= width; ++w)
    {
      vector_3d vert{ scale * w, scale * h, 0.0 };
      verts->push_back(origin + vert);
      if (w > 0 && h > 0)
      {
        unsigned prev_x = index - 1;
        unsigned prev_y = index - width - 1;
        unsigned prev_xy = prev_y - 1;
        faces->push_back({index, prev_xy, prev_y});
        faces->push_back({index, prev_x, prev_xy});
      }
      ++index;
    }
  }

  return std::make_shared<mesh>(std::unique_ptr<mesh_vertex_array_base>(verts),
                                std::unique_ptr<mesh_face_array_base>(faces));
}

// construct map of landmarks will all locations at c
kwiver::vital::landmark_map_sptr
init_landmarks( kwiver::vital::landmark_id_t num_lm,
                const kwiver::vital::vector_3d& c = kwiver::vital::vector_3d(0, 0, 0) )
{
  using namespace kwiver::vital;

  landmark_map::map_landmark_t lm_map;
  for ( landmark_id_t i = 0; i < num_lm; ++i )
  {
    lm_map[i] = landmark_sptr( new landmark_d( c ) );
  }
  return landmark_map_sptr( new simple_landmark_map( lm_map ) );
}

// add Gaussian noise to the landmark positions
kwiver::vital::landmark_map_sptr
noisy_landmarks( kwiver::vital::landmark_map_sptr  landmarks,
                 double                     stdev = 1.0 )
{
  using namespace kwiver::vital;

  landmark_map::map_landmark_t lm_map = landmarks->landmarks();
  for( landmark_map::map_landmark_t::value_type& p : lm_map )
  {
    landmark_sptr l = p.second->clone();
    landmark_d& lm = dynamic_cast<landmark_d&>(*l);

    lm.set_loc( lm.get_loc() + random_point3d( stdev ) );
    lm_map[p.first] = l;
  }
  return landmark_map_sptr( new simple_landmark_map( lm_map ) );
}

// create a camera sequence (elliptical path)
kwiver::vital::camera_map_sptr
camera_seq(kwiver::vital::frame_id_t num_cams,
           kwiver::vital::camera_intrinsics_sptr K,
           double scale = 1.0,
           double degree_span = 115)
{
  using namespace kwiver::vital;
  camera_map::map_camera_t cameras;
  const double angle = degree_span * deg_to_rad;

  // create a camera sequence (elliptical path)
  rotation_d R; // identity
  for ( frame_id_t i = 0; i < num_cams; ++i )
  {
    double frac = static_cast< double > ( i ) / num_cams;
    double x = 4 * std::cos( angle * frac );
    double y = 3 * std::sin( angle * frac );
    simple_camera_perspective* cam =
      new simple_camera_perspective(scale * vector_3d(x,y,2+frac), R, K);
    // look at the origin
    cam->look_at( vector_3d( 0, 0, 0 ) );
    cameras[i] = camera_sptr( cam );
  }
  return camera_map_sptr( new simple_camera_map( cameras ) );
}

// create a camera sequence (elliptical path)
kwiver::vital::camera_map_sptr
camera_seq(kwiver::vital::frame_id_t num_cams = 20,
           kwiver::vital::simple_camera_intrinsics K =
           kwiver::vital::simple_camera_intrinsics(
             1000, { 640, 480 }, 1.0, 0.0, {}, 1280, 960),
           double scale = 1.0,
           double degree_span = 115)
{
  return camera_seq(num_cams, K.clone(), scale, degree_span);
}

// create an initial camera sequence with all cameras at the same location
kwiver::vital::camera_map_sptr
init_cameras(kwiver::vital::frame_id_t num_cams,
             kwiver::vital::camera_intrinsics_sptr K)
{
  using namespace kwiver::vital;
  camera_map::map_camera_t cameras;

  // create a camera sequence (elliptical path)

  rotation_d R; // identity
  vector_3d c( 0, 0, 1 );
  for ( frame_id_t i = 0; i < num_cams; ++i )
  {
    simple_camera_perspective* cam = new simple_camera_perspective(c, R, K);
    // look at the origin
    cam->look_at( vector_3d( 0, 0, 0 ), vector_3d( 0, 1, 0 ) );
    cameras[i] = camera_sptr( cam );
  }
  return camera_map_sptr( new simple_camera_map( cameras ) );
}

// create an initial camera sequence with all cameras at the same location
kwiver::vital::camera_map_sptr
init_cameras(kwiver::vital::frame_id_t num_cams = 20,
             kwiver::vital::simple_camera_intrinsics K =
                 kwiver::vital::simple_camera_intrinsics(1000, kwiver::vital::vector_2d(640, 480)))
{
  return init_cameras(num_cams, K.clone());
}

// add positional and rotational Gaussian noise to cameras
kwiver::vital::camera_map_sptr
noisy_cameras( kwiver::vital::camera_map_sptr cameras,
               double pos_stdev = 1.0, double rot_stdev = 1.0 )
{
  using namespace kwiver::vital;

  camera_map::map_camera_t cam_map;
  for( camera_map::map_camera_t::value_type const& p : cameras->cameras() )
  {
    auto cam_ptr = std::dynamic_pointer_cast<vital::camera_perspective>(p.second);
    auto c = std::dynamic_pointer_cast<vital::camera_perspective>(cam_ptr->clone());

    simple_camera_perspective& cam =
      dynamic_cast<simple_camera_perspective&>(*c);

    cam.set_center( cam.get_center() + random_point3d( pos_stdev ) );
    rotation_d rand_rot( random_point3d( rot_stdev ) );
    cam.set_rotation( cam.get_rotation() * rand_rot );

    cam_map[p.first] = c;
  }
  return camera_map_sptr( new simple_camera_map( cam_map ) );
}

// randomly drop a fraction of the track states
kwiver::vital::feature_track_set_sptr
subset_tracks( kwiver::vital::feature_track_set_sptr in_tracks, double keep_frac = 0.75 )
{
  using namespace kwiver::vital;

  std::srand( 0 );
  std::vector< track_sptr > tracks = in_tracks->tracks();
  std::vector< track_sptr > new_tracks;
  const int rand_thresh = static_cast< int > ( keep_frac * RAND_MAX );
  for( const track_sptr &t : tracks )
  {
    auto nt = track::create();

    nt->set_id( t->id() );
    std::cout << "track " << t->id() << ":";
    for( auto const& ts : *t )
    {
      if ( std::rand() < rand_thresh )
      {
        nt->append( ts->clone() );
        std::cout << " .";
      }
      else
      {
        std::cout << " X";
      }
    }
    std::cout << std::endl;
    new_tracks.push_back( nt );
  }
  return std::make_shared<feature_track_set>( new_tracks );
}

// add Gaussian noise to track feature locations
kwiver::vital::feature_track_set_sptr
noisy_tracks( kwiver::vital::feature_track_set_sptr in_tracks, double stdev = 1.0 )
{
  using namespace kwiver::vital;

  std::vector< track_sptr > tracks = in_tracks->tracks();
  std::vector< track_sptr > new_tracks;
  for( const track_sptr &t : tracks )
  {
    auto nt = track::create();
    nt->set_id(t->id());
    for(track::history_const_itr it=t->begin(); it!=t->end(); ++it)
    {
      auto fts = std::dynamic_pointer_cast<feature_track_state>(*it);
      if( !fts || !fts->feature )
      {
        continue;
      }
      vector_2d loc = fts->feature->loc() + random_point2d(stdev);
      auto new_fts = std::make_shared<feature_track_state>(*fts);
      new_fts->feature = std::make_shared<feature_d>(loc);
      nt->append(new_fts);
    }
    new_tracks.push_back(nt);
  }
  return std::make_shared<feature_track_set>( new_tracks );
}

// randomly select a fraction of the track states to make outliers
// outliers are created by adding random noise with large standard deviation
kwiver::vital::feature_track_set_sptr
add_outliers_to_tracks(kwiver::vital::feature_track_set_sptr in_tracks,
                       double outlier_frac=0.1,
                       double stdev=20.0)
{
  using namespace kwiver::vital;

  std::srand(0);
  std::vector<track_sptr> tracks = in_tracks->tracks();
  std::vector<track_sptr> new_tracks;
  const int rand_thresh = static_cast<int>(outlier_frac * RAND_MAX);
  for(const track_sptr& t : tracks)
  {
    track_sptr nt = track::create();
    nt->set_id( t->id() );
    for( const auto &ts : *t )
    {
      auto fts = std::dynamic_pointer_cast<feature_track_state>(ts);
      if( !fts || !fts->feature )
      {
        std::cout << " X";
        continue;
      }
      if(std::rand() < rand_thresh)
      {
        vector_2d loc = fts->feature->loc() + random_point2d( stdev );
        auto new_fts = std::make_shared<feature_track_state>(*fts);
        new_fts->feature = std::make_shared<feature_d>(loc);
        nt->append( new_fts );
        std::cout << " M";
      }
      else
      {
        std::cout << " .";
        nt->append(ts->clone());
      }
    }
    std::cout << std::endl;
    new_tracks.push_back( nt );
  }
  return std::make_shared<feature_track_set>( new_tracks );
}

// set inlier state on all track states
void
reset_inlier_flag( kwiver::vital::feature_track_set_sptr tracks,
                   bool target_state=false )
{
  using namespace kwiver::vital;

  for( track_sptr t : tracks->tracks() )
  {
    for( auto fts : *t | as_feature_track )
    {
      if( !fts )
      {
        continue;
      }
      fts->inlier = target_state;
    }
  }
}

} // end namespace testing
} // end namespace kwiver

#endif // VITAL_TEST_TEST_SCENE_H_
