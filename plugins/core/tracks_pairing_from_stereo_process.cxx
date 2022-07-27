/**
 * \file
 * \brief Compute object tracks pair from stereo depth map information
 */

#include "tracks_pairing_from_stereo_process.h"

#include <vital/vital_types.h>
#include <vital/types/image_container.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/object_track_set.h>
#include <vital/types/camera_perspective_map.h>
#include <vital/types/bounding_box.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <arrows/ocv/image_container.h>

#include <sprokit/processes/kwiver_type_traits.h>

namespace kv = kwiver::vital;


namespace viame
{

namespace core
{

create_config_trait( cameras_directory, std::string, "",
                     "The calibrated cameras files directory" );

create_port_trait( object_track_set1, object_track_set, "Set of object tracks1." );
create_port_trait( object_track_set2, object_track_set, "Set of object tracks2." );
create_port_trait( filtered_object_track_set1, object_track_set, "The stereo filtered object tracks1." );
create_port_trait( filtered_object_track_set2, object_track_set, "The stereo filtered object tracks2." );


// =============================================================================
// Private implementation class
class tracks_pairing_from_stereo_process::priv
{
public:
  explicit priv( tracks_pairing_from_stereo_process* parent );
  ~priv();

  // Configuration settings
  std::string m_cameras_directory;
  kv::camera_map::map_camera_t m_cameras;

  kv::logger_handle_t m_logger;

  // Other variables
  tracks_pairing_from_stereo_process* parent;

  // Camera depth informations
  cv::Mat m_Q, m_K1, m_D1, m_R1, m_P1;

  // Tracks status
  // TODO: FORGET TRACKS AT SOME POINT
  std::map<int, kv::track_sptr > m_tracks_with_3d_left;

  void
  load_camera_calibration()
  {
    try {
      auto intrinsics_path = m_cameras_directory + "/intrinsics.yml";
      auto extrinsics_path = m_cameras_directory + "/extrinsics.yml";
      auto fs = cv::FileStorage(intrinsics_path, cv::FileStorage::Mode::READ);

      // load left camera intrinsic parameters
      if (!fs.isOpened())
        VITAL_THROW(kv::file_not_found_exception, intrinsics_path, "could not locate file in path");

      fs["M1"] >> m_K1;
      fs["D1"] >> m_D1;
      fs.release();

      // load extrinsic parameters
      fs = cv::FileStorage(extrinsics_path, cv::FileStorage::Mode::READ);
      if (!fs.isOpened())
        VITAL_THROW(kv::file_not_found_exception, extrinsics_path, "could not locate file in path");

      fs["Q"] >> m_Q;
      fs["R1"] >> m_R1;
      fs["P1"] >> m_P1;
      fs.release();
    }
    catch ( const kv::file_not_found_exception& e)
    {
      VITAL_THROW( kv::invalid_data, "Calibration file not found : " + std::string(e.what()) );
    }
  }

  float
  compute_median
  (std::vector<float> values,
   bool is_sorted=0)
  {
    float median = 0;
    size_t size = values.size();
    if (size > 0)
    {
      if (!is_sorted)
        std::sort(values.begin(), values.end());

      if (size % 2 == 0)
        median = (values[size / 2 - 1] + values[size / 2]) / 2;
      else
        median = values[size / 2];
    }
    return median;
  }

  void
  get_bbox_3d_position_size_left
  (cv::Point2f& bbox_center,
   cv::Size& bbox_size,
   const kv::bounding_box_d bbox)
  {
    // undistort bbox corners + center
    cv::Mat_<cv::Point2f> points_raw(1, 5);
    points_raw(0) = cv::Point2f(bbox.center().x(), bbox.center().y());
    points_raw(1) = cv::Point2f(bbox.min_x(), bbox.min_y()); // upper left
    points_raw(2) = cv::Point2f(bbox.min_x(), bbox.max_y()); // bottom left
    points_raw(3) = cv::Point2f(bbox.max_x(), bbox.min_y());  // upper right
    points_raw(4) = cv::Point2f(bbox.max_x(), bbox.max_y());  // bottom right

    cv::Mat_<cv::Point2f> points_undist(1, 5);
    cv::undistortPoints(points_raw, points_undist,
                        m_K1, m_D1, m_R1, m_P1);

    // As the undistorted corners may not be an axis-aligned box, estimate the
    // height and width from average measures
    bbox_size = cv::Size(((points_undist(2).y -  points_undist(1).y)
                      + (points_undist(4).y -  points_undist(3).y)) / 2,
                          ((points_undist(3).x -  points_undist(1).x)
                      + (points_undist(4).x -  points_undist(2).x)) / 2);

    bbox_center = points_undist(0);

  }

  float  // returns a score
  estimate_3d_position_from_left_bbox
  (cv::Point3f& point3d,
   const cv::Point2f& bbox_center,
   const cv::Size& bbox_size,
   const cv::Mat& pos_3d_map)
  {
    // depth from median of values in the center part of the bounding box
    float ratio = 1./3;
    float crop_width = ratio * bbox_size.width;
    float crop_height = ratio * bbox_size.height;
    cv::Rect crop_rect{(int)(bbox_center.x - crop_width/2),
                       (int)(bbox_center.y - crop_height/2),
                       (int)crop_width, (int)crop_height};

    // Intersect crop rectangle with 3D map rect to avoid out of bounds crop
    crop_rect = crop_rect & cv::Rect(0, 0, pos_3d_map.size().width, pos_3d_map.size().height);
    auto crop = pos_3d_map(crop_rect);

    // Compute medians in cropped patch
    cv::Mat channels[3];
    cv::split(crop, channels);
    cv::Mat sort_indices;

    // Select for valid points (with z > 0 and z != inf ) and compute xs, ys, zs
    // median from those
    cv::Mat xs = channels[0].reshape(1, 1);
    cv::Mat ys = channels[1].reshape(1, 1);
    cv::Mat zs = channels[2].reshape(1, 1);
    int nb_val = zs.size().width;
    if (zs.depth() != CV_32F)
    {
      throw std::runtime_error( "depth values are not of type cv::CV_32F." );
    }

    std::vector<bool> is_valid(nb_val);
    for (int i=0; i< nb_val; i++)
    {
      is_valid[i] = ((zs.at<float>(i) > 0)
        && std::isfinite(xs.at<float>(i))
        && std::isfinite(ys.at<float>(i))
        && std::isfinite(zs.at<float>(i))
        );
    }

    size_t nb_valid = std::count(is_valid.begin(), is_valid.end() , true);

    float x, y, z;
    if (nb_valid > 0)
    {
      std::vector<float> valid_xs(nb_valid), valid_ys(nb_valid), valid_zs(nb_valid);

      int next_valid_idx = 0;
      for( int i = 0; i < nb_val; i++)
      {
        if (is_valid[i])
        {
          valid_xs[next_valid_idx] = xs.at<float>(i);
          valid_ys[next_valid_idx] = ys.at<float>(i);
          valid_zs[next_valid_idx] = zs.at<float>(i);
          next_valid_idx++;
        }
      }

      x = compute_median(valid_xs);
      y = compute_median(valid_ys);
      z = compute_median(valid_zs);

    }
    else
    {
      x = 0;
      y = 0;
      z = 0;
    }
    point3d = cv::Point3f(x, y, z);

    return ((float)nb_valid) / (crop_width * crop_height);

  }
};


// -----------------------------------------------------------------------------
tracks_pairing_from_stereo_process::priv
::priv( tracks_pairing_from_stereo_process* ptr )
  : m_cameras_directory( "" )
  , parent( ptr )
{
}


tracks_pairing_from_stereo_process::priv
::~priv()
{
}


// =============================================================================
tracks_pairing_from_stereo_process
::tracks_pairing_from_stereo_process( kv::config_block_sptr const& config )
  : process( config ),
    d( new tracks_pairing_from_stereo_process::priv( this ) )
{
  make_ports();
  make_config();
}


tracks_pairing_from_stereo_process
::~tracks_pairing_from_stereo_process()
{
}


// -----------------------------------------------------------------------------
void
tracks_pairing_from_stereo_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- inputs --
  declare_input_port_using_trait( image, optional );
  declare_input_port_using_trait( timestamp, optional );
  declare_input_port_using_trait( object_track_set1, required );
  declare_input_port_using_trait( object_track_set2, required );
  declare_input_port_using_trait( depth_map, required );

  // -- outputs --
  declare_output_port_using_trait( timestamp, optional );
  // declare_output_port_using_trait( object_track_set, optional );
  declare_output_port_using_trait( filtered_object_track_set1, optional );
  declare_output_port_using_trait( filtered_object_track_set2, optional );
}

// -----------------------------------------------------------------------------
void
tracks_pairing_from_stereo_process
::make_config()
{
  declare_config_using_trait( cameras_directory );
}

// -----------------------------------------------------------------------------
void
tracks_pairing_from_stereo_process
::_configure()
{
  d->m_cameras_directory = config_value_using_trait( cameras_directory );
  d->load_camera_calibration();
}

// -----------------------------------------------------------------------------
void tracks_pairing_from_stereo_process::_step() {
  kv::object_track_set_sptr input_tracks1, input_tracks2;
  kv::image_container_sptr depth_map;
  kv::timestamp timestamp;

  input_tracks1 = grab_from_port_using_trait(object_track_set1);
  input_tracks2 = grab_from_port_using_trait(object_track_set2);

  if (has_input_port_edge_using_trait(timestamp)) {
    timestamp = grab_from_port_using_trait(timestamp);
  }
  if (has_input_port_edge_using_trait(depth_map)) {
    depth_map = grab_from_port_using_trait(depth_map);
  }

  // convert disparity map to opencv for further computing
  cv::Mat cv_disparity = kwiver::arrows::ocv::image_container::vital_to_ocv(depth_map->get_image(),
                                                                            kwiver::arrows::ocv::image_container::BGR_COLOR);

  // Disparity to 3d map
  cv::Mat cv_pos_3d_map;
  cv::reprojectImageTo3D(cv_disparity, cv_pos_3d_map, d->m_Q, false);

  std::vector<kv::track_sptr> filtered_tracks1, filtered_tracks2;

  for (const auto &track1: input_tracks1->tracks()) {
    // Check if a 3d track already exists with this id in order to update it
    // instead of generating a new track
    kv::track_sptr track_3d_left;
    auto t_ptr = d->m_tracks_with_3d_left.find(track1->id());
    if (t_ptr != d->m_tracks_with_3d_left.end()) {
      track_3d_left = t_ptr->second;
    } else {
      track_3d_left = track1;
      d->m_tracks_with_3d_left.insert({track1->id(), track_3d_left});
    }

    // Replace tracks that are present without having a current state
    // TODO: check that there is no change in the tracks
    if (track1->last_frame() < timestamp.get_frame()) {
      filtered_tracks1.push_back(track_3d_left);
    }

    // Vanilla implem:
    // At each frame, compute 3d position for the bounding boxes on the left
    // camera and add this value to the output track
    for (auto state1: *track1 | kv::as_object_track) {
      // run only for current frame (tracks with ho current frame have already
      // been forwarded)
      if (timestamp.get_frame() == state1->frame()) {
        // If not a new track, add state to existing track
        if (state1->frame() > track_3d_left->last_frame()) {
          track_3d_left->append(state1);
        }

        kv::bounding_box_d bbox1 = state1->detection()->bounding_box();
        cv::Point2f bbox1_center;
        cv::Size bbox1_size;
        d->get_bbox_3d_position_size_left(bbox1_center, bbox1_size, bbox1);
        cv::Point3f center3d;
        float score = d->estimate_3d_position_from_left_bbox(center3d, bbox1_center, bbox1_size, cv_pos_3d_map);

        // Add 3d estimations
        if (score > 0) {
          state1->detection()->add_note(":x=" + std::to_string(center3d.x));
          state1->detection()->add_note(":y=" + std::to_string(center3d.y));
          state1->detection()->add_note(":z=" + std::to_string(center3d.z));
          state1->detection()->add_note(":score=" + std::to_string(score));
        }
        filtered_tracks1.push_back(track_3d_left);
      }
    }
  }

  kv::object_track_set_sptr output1(new kv::object_track_set(filtered_tracks1));
  kv::object_track_set_sptr output2(new kv::object_track_set(filtered_tracks2));

  push_to_port_using_trait(timestamp, timestamp);
  push_to_port_using_trait(filtered_object_track_set1, output1);
  push_to_port_using_trait(filtered_object_track_set2, output2);
}

} // end namespace core

} // end namespace viame
