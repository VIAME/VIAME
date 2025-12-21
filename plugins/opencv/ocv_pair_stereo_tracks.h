#ifndef VIAME_OCV_PAIR_STEREO_TRACKS_H
#define VIAME_OCV_PAIR_STEREO_TRACKS_H

#include <vital/types/timestamp.h>
#include <vital/types/object_track_set.h>
#include <plugins/opencv/viame_opencv_export.h>

namespace cv {
class Mat;
}

namespace viame {

class ocv_pair_stereo_detections;

struct Detections3DPositions;

/// @brief Structure storing the different frames left and right tracks have been paired at
struct IdPair
{
  kwiver::vital::track_id_t left_id;
  kwiver::vital::track_id_t right_id;
};

struct VIAME_OPENCV_EXPORT Pairing
{
  std::set< kwiver::vital::frame_id_t > frame_set;
  IdPair left_right_id_pair;
};


/// @brief Class responsible for the tracks stereo pairing logic
/// Uses camera calibration information, left and right tracks and disparity map to find corresponding tracks from left
/// To right.
///
/// Instance keeps track of the different tracks seen in left and right camera.
/// Otherwise, class is meant to be used as a toolkit for pairing the two track feeds.
class VIAME_OPENCV_EXPORT ocv_pair_stereo_tracks
{
  const std::shared_ptr< ocv_pair_stereo_detections > m_detection_pairing;

public:

  ocv_pair_stereo_tracks();

  // Configuration settings
  std::string m_cameras_directory;
  kwiver::vital::logger_handle_t m_logger;
  float m_iou_pair_threshold{ 0.1f };
  int m_min_detection_number_threshold{ 0 };
  int m_max_detection_number_threshold{ std::numeric_limits< int >::max() };
  int m_min_detection_surface_threshold_pix{ 0 };
  int m_max_detection_surface_threshold_pix{ std::numeric_limits< int >::max() };
  int m_detection_split_threshold{ 0 };
  bool m_do_split_detections{ false };
  std::string m_pairing_method{ "PAIRING_3D" };
  bool m_verbose{}; // Set true to activate debug print

  // Tracks status memo
  std::map< kwiver::vital::track_id_t, kwiver::vital::track_sptr > m_tracks_with_3d_left, m_right_tracks_memo;

  // Track ID pairing
  std::map< size_t, Pairing > m_left_to_right_pairing;


  /// @brief Load matrix calibration from settings camera directory
  void load_camera_calibration();

  /// @brief Update 3D tracks positions given a list of tracks and tracks disparity map
  std::tuple< std::vector< kwiver::vital::track_sptr >, std::vector< viame::Detections3DPositions > >
  update_left_tracks_3d_position( const std::vector< kwiver::vital::track_sptr >& tracks,
                                  const cv::Mat& cv_disparity_map,
                                  const kwiver::vital::timestamp& timestamp );

  /// @brief Filter input tracks to keep only tracks with detection in the current frame.
  std::vector< kwiver::vital::track_sptr >
  keep_right_tracks_in_current_frame( const std::vector< kwiver::vital::track_sptr >& tracks,
                                      const kwiver::vital::timestamp& timestamp );

  /// @brief returns last track id available in both left and right track map
  kwiver::vital::track_id_t last_left_right_track_id() const;

  /// @brief Update left and right tracks pairs using the currently set pairing method.
  ///     If pairing is set to 3D, uses @ref pair_left_right_tracks_using_3d_center.
  ///     Otherwise, uses @ref pair_left_right_tracks_using_bbox_iou.
  void pair_left_right_tracks( const std::vector< kwiver::vital::track_sptr >& left_tracks,
                               const std::vector< viame::Detections3DPositions >& left_3d_pos,
                               const std::vector< kwiver::vital::track_sptr >& right_tracks,
                               const kwiver::vital::timestamp& timestamp );

  /// @brief Apply pairing to dict of left and right tracks and pair left and right tracks given accumulated information
  std::tuple< std::vector< kwiver::vital::track_sptr >, std::vector< kwiver::vital::track_sptr > >
  get_left_right_tracks_with_pairing();

  /// @brief Remove tracks in input list that don't match the min / max detection and min / max surface thresholds
  std::vector< kwiver::vital::track_sptr >
  filter_tracks_with_threshold( std::vector< kwiver::vital::track_sptr > tracks ) const;

  /// @brief Cantor pairing function mapping N x N -> N
  /// Allow to map pairs of left / right ids to one unique natural integer
  static size_t cantor_pairing( size_t i, size_t j )
  {
    return ( ( i + j ) * ( i + j + 1u ) ) / 2u + j;
  }

  void append_paired_frame( const kwiver::vital::track_sptr& left_track,
                            const kwiver::vital::track_sptr& right_track,
                            const kwiver::vital::timestamp& timestamp );

  /// @brief Splits the input tracks to new tracks if left / right detection pairing jumps between frames.
  std::tuple< std::set< kwiver::vital::track_id_t >, std::set< kwiver::vital::track_id_t > >
  split_paired_tracks_to_new_tracks( std::vector< kwiver::vital::track_sptr >& left_tracks,
                                     std::vector< kwiver::vital::track_sptr >& right_tracks );

  /// @brief Returns the most likely left / right track pairing across the different frames.
  std::tuple< std::set< kwiver::vital::track_id_t >, std::set< kwiver::vital::track_id_t > >
  select_most_likely_pairing( std::vector< kwiver::vital::track_sptr >& left_tracks,
                              std::vector< kwiver::vital::track_sptr >& right_tracks );

  /// @class Range
  /// @brief Container struct to pair left / right tracks ids with associated first / last frames ranges.
  struct Range
  {
    kwiver::vital::track_id_t left_id, right_id, new_track_id;
    kwiver::vital::frame_id_t frame_id_first, frame_id_last;
    int detection_count;
  };

  /// @brief Creates ranges from input frame / pairing list corresponding to coherent detection pairings.
  std::vector< viame::ocv_pair_stereo_tracks::Range >
  create_split_ranges_from_track_pairs( const std::map< size_t, Pairing >& source_range ) const;

  /// @brief Uses input slit ranges to create new left / rigth tracks with associated track ids.
  std::tuple< std::set< kwiver::vital::track_id_t >, std::set< kwiver::vital::track_id_t > >
  split_ranges_to_tracks( const std::vector< Range >& ranges,
                          std::vector< kwiver::vital::track_sptr >& left_tracks,
                          std::vector< kwiver::vital::track_sptr >& right_tracks );
};

} // viame

#endif // VIAME_OCV_PAIR_STEREO_TRACKS_H
