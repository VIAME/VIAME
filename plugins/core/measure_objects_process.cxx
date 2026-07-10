/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Stereo measurement process implementation
 */

#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <set>

#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/object_track_set.h>
#include <vital/types/image_container.h>
#include <vital/types/vector.h>
#include <vital/types/point.h>
#include <vital/types/bounding_box.h>
#include <vital/types/track.h>
#include <vital/util/string.h>

#include <sprokit/processes/kwiver_type_traits.h>

#include "measure_objects_process.h"
#include "measurement_utilities.h"
#include "pair_stereo_detections.h"
#include "pair_stereo_tracks.h"
#include "camera_rig_io.h"

namespace kv = kwiver::vital;

namespace viame
{

namespace core
{

// Only calibration_file and min_track_states are process-specific;
// rest comes from map_keypoints_to_camera_settings
create_config_trait( calibration_file, std::string, "",
  "Input filename for the calibration file to use" );

create_config_trait( min_track_states, unsigned, "0",
  "Minimum number of track states (summed across all cameras) required before "
  "a track is included in the output. A detection in one camera counts as 1, "
  "in two cameras as 2. Measurement is still performed on every frame; this "
  "only controls which tracks appear in the output. "
  "Set to 0 to disable filtering (default)." );

create_config_trait( max_stereo_rms, double, "-1.0",
  "Maximum acceptable stereo-triangulation RMS reprojection error (pixels) "
  "for a per-frame left/right track pairing to be recorded in the pairing "
  "accumulator. Pairs whose measurement rms exceeds this are dropped from "
  "the union-find at resolve time — this rejects false same-tracker-ID "
  "and keypoint-projection matches whose triangulation did not converge. "
  "Set to <= 0 to disable filtering. Typical value: 20.0." );

create_config_trait( max_bbox_y_center_offset, double, "-1.0",
  "Maximum |y_center_left - y_center_right| (pixels) for a per-frame left/"
  "right track pairing to be recorded in the pairing accumulator. For a "
  "near-parallel stereo rig the y-center offset equals the epipolar "
  "residual, so this rejects pairs of unrelated fish at different image "
  "heights even when their triangulation rms happens to be small. Kept as "
  "absolute pixels because detection/keypoint noise is roughly constant "
  "per-pixel — it does not scale with bbox size, so normalizing by height "
  "over-rejects small targets. Set to <= 0 to disable. Typical value: 30." );

create_config_trait( max_bbox_area_ratio, double, "-1.0",
  "Maximum ratio max(L_area, R_area)/min(L_area, R_area) for a per-frame "
  "pairing. Rejects pairs whose bounding boxes differ too much in area "
  "(typical cause: coincidental same-tracker-ID matches between a small "
  "fish in one camera and a large fish or school in the other). Set to "
  "<= 0 to disable. Typical value: 3.0." );

create_port_trait( object_track_set1, object_track_set,
  "The stereo filtered object tracks1.")
create_port_trait( object_track_set2, object_track_set,
  "The stereo filtered object tracks2.")
create_port_trait( disparity_image, image,
  "Disparity map (input for external_disparity method, output from compute_disparity method).")
create_port_trait( rectified_left_image, image,
  "Output rectified left image.")
create_port_trait( rectified_right_image, image,
  "Output rectified right image.")

// =============================================================================
// Private implementation class
class measure_objects_process::priv
{
public:
  explicit priv( measure_objects_process* parent );
  ~priv();

  // Process-specific config
  std::string m_calibration_file;
  unsigned m_min_track_states;
  double m_max_stereo_rms;
  double m_max_bbox_y_center_offset;
  double m_max_bbox_area_ratio;

  // Measurement settings (contains all algo parameters and algorithm pointers)
  map_keypoints_to_camera_settings m_settings;

  // Parsed matching methods (derived from settings)
  std::vector< std::string > m_matching_methods;

  // Other variables
  kv::camera_rig_stereo_sptr m_calibration;
  unsigned m_frame_counter;
  std::set< std::string > p_port_list;
  measure_objects_process* parent;

  // Measurement utilities
  map_keypoints_to_camera m_utilities;

  // Track pairing (shared with pair_stereo_detections_process)
  stereo_track_pairer m_track_pairer;

  // Persistent right tracks created for left-only detections across frames
  std::map< kv::track_id_t, kv::track_sptr > m_created_right_tracks;

  // Detection pairing map: synthetic_right_id (= left_id) → actual_right_id
  // Used to link actual right camera tracks in the union-find so that
  // transitive linking works when a left track dies and a new one starts.
  std::map< kv::track_id_t, kv::track_id_t > m_detection_paired_right_ids;
};


// -----------------------------------------------------------------------------
measure_objects_process::priv
::priv( measure_objects_process* ptr )
  : m_calibration_file( "" )
  , m_min_track_states( 0 )
  , m_max_stereo_rms( -1.0 )
  , m_max_bbox_y_center_offset( -1.0 )
  , m_max_bbox_area_ratio( -1.0 )
  , m_calibration()
  , m_frame_counter( 0 )
  , parent( ptr )
{
}


measure_objects_process::priv
::~priv()
{
}

// =============================================================================
measure_objects_process
::measure_objects_process( kv::config_block_sptr const& config )
  : process( config ),
    d( new measure_objects_process::priv( this ) )
{
  this->set_data_checking_level( check_none );

  make_ports();
  make_config();
}


measure_objects_process
::~measure_objects_process()
{
}


// -----------------------------------------------------------------------------
void
measure_objects_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- inputs --
  declare_input_port_using_trait( timestamp, optional );
  declare_input_port_using_trait( disparity_image, optional );

  // -- outputs --
  declare_output_port_using_trait( object_track_set1, required );
  declare_output_port_using_trait( object_track_set2, optional );
  declare_output_port_using_trait( timestamp, optional );
  declare_output_port_using_trait( disparity_image, optional );
  declare_output_port_using_trait( rectified_left_image, optional );
  declare_output_port_using_trait( rectified_right_image, optional );
}

// -----------------------------------------------------------------------------
void
measure_objects_process
::make_config()
{
  // Process-specific config
  declare_config_using_trait( calibration_file );
  declare_config_using_trait( min_track_states );
  declare_config_using_trait( max_stereo_rms );
  declare_config_using_trait( max_bbox_y_center_offset );
  declare_config_using_trait( max_bbox_area_ratio );

  // Merge in map_keypoints_to_camera_settings configuration
  kv::config_block_sptr settings_config = d->m_settings.get_configuration();
  for( auto const& key : settings_config->available_values() )
  {
    declare_configuration_key(
      key,
      settings_config->get_value< std::string >( key ),
      settings_config->get_description( key ) );
  }

  // Merge in stereo track pairer configuration
  kv::config_block_sptr tp_config = d->m_track_pairer.get_configuration();
  for( auto const& key : tp_config->available_values() )
  {
    declare_configuration_key(
      key,
      tp_config->get_value< std::string >( key ),
      tp_config->get_description( key ) );
  }
}

// -----------------------------------------------------------------------------
void
measure_objects_process
::_configure()
{
  // Get process-specific config
  d->m_calibration_file = config_value_using_trait( calibration_file );
  d->m_min_track_states = config_value_using_trait( min_track_states );
  d->m_max_stereo_rms = config_value_using_trait( max_stereo_rms );
  d->m_max_bbox_y_center_offset = config_value_using_trait( max_bbox_y_center_offset );
  d->m_max_bbox_area_ratio = config_value_using_trait( max_bbox_area_ratio );

  if( d->m_calibration_file.empty() )
  {
    LOG_ERROR( logger(), "Calibration file not specified" );
    throw std::runtime_error( "Calibration file not specified" );
  }

  d->m_calibration = viame::read_stereo_rig( d->m_calibration_file );

  if( !d->m_calibration || !d->m_calibration->left() || !d->m_calibration->right() )
  {
    LOG_ERROR( logger(), "Failed to load calibration file: " + d->m_calibration_file );
    throw std::runtime_error( "Failed to load calibration file: " + d->m_calibration_file );
  }

  // Configure settings from config block
  d->m_settings.set_configuration( get_config() );

  // Configure track pairer
  d->m_track_pairer.set_configuration( get_config() );

  // Validate matching methods
  std::string validation_error = d->m_settings.validate_matching_methods();
  if( !validation_error.empty() )
  {
    LOG_ERROR( logger(), validation_error );
    throw std::runtime_error( validation_error );
  }

  // Get parsed methods
  d->m_matching_methods = d->m_settings.get_matching_methods();

  LOG_INFO( logger(), "Matching methods (in order): " + d->m_settings.matching_methods );

  // Configure utilities from settings
  d->m_utilities.configure( d->m_settings );

  // Check for warnings about feature algorithms
  auto warnings = d->m_settings.check_feature_algorithm_warnings();
  for( const auto& warning : warnings )
  {
    LOG_WARN( logger(), warning );
  }
}

// ----------------------------------------------------------------------------
void
measure_objects_process
::_init()
{
  this->set_data_checking_level( check_none );
}

// ----------------------------------------------------------------------------
void
measure_objects_process
::input_port_undefined( port_t const& port_name )
{
  LOG_TRACE( logger(), "Processing undefined input port: \"" << port_name << "\"" );

  if( !kv::starts_with( port_name, "_" ) )
  {
    if( d->p_port_list.count( port_name ) == 0 )
    {
      port_flags_t required;
      required.insert( flag_required );

      if( port_name.find( "image" ) != std::string::npos )
      {
        declare_input_port(
          port_name,
          image_port_trait::type_name,
          required,
          "image container input" );
      }
      else
      {
        declare_input_port(
          port_name,
          object_track_set_port_trait::type_name,
          required,
          "object track set input" );
      }

      d->p_port_list.insert( port_name );
    }
  }
}

// -----------------------------------------------------------------------------
void
measure_objects_process
::_finalize()
{
  // Emit the resolved unified track sets before pushing complete datums,
  // so downstream writers see the final data prior to end-of-stream.
  if( d->m_track_pairer.accumulation_enabled() )
  {
    std::vector< kv::track_sptr > output_trks1, output_trks2;
    d->m_track_pairer.resolve_accumulated_pairings( output_trks1, output_trks2 );

    if( d->m_min_track_states > 0 )
    {
      std::map< kv::track_id_t, unsigned > state_counts;
      for( const auto& trk : output_trks1 )
        state_counts[ trk->id() ] += static_cast< unsigned >( trk->size() );
      for( const auto& trk : output_trks2 )
        state_counts[ trk->id() ] += static_cast< unsigned >( trk->size() );

      auto meets = [&]( const kv::track_sptr& t )
      {
        auto it = state_counts.find( t->id() );
        return it != state_counts.end() && it->second >= d->m_min_track_states;
      };
      output_trks1.erase(
        std::remove_if( output_trks1.begin(), output_trks1.end(),
                        [&]( const kv::track_sptr& t ){ return !meets( t ); } ),
        output_trks1.end() );
      output_trks2.erase(
        std::remove_if( output_trks2.begin(), output_trks2.end(),
                        [&]( const kv::track_sptr& t ){ return !meets( t ); } ),
        output_trks2.end() );
    }

    LOG_INFO( logger(), "Resolved " << output_trks1.size() << " left and "
              << output_trks2.size() << " right accumulated tracks" );

    push_to_port_using_trait(
      object_track_set1, std::make_shared< kv::object_track_set >( output_trks1 ) );
    push_to_port_using_trait(
      object_track_set2, std::make_shared< kv::object_track_set >( output_trks2 ) );
  }

  mark_process_as_complete();

  const sprokit::datum_t dat = sprokit::datum::complete_datum();

  push_datum_to_port_using_trait( object_track_set1, dat );
  push_datum_to_port_using_trait( object_track_set2, dat );
  push_datum_to_port_using_trait( timestamp, dat );
  push_datum_to_port_using_trait( disparity_image, dat );
  push_datum_to_port_using_trait( rectified_left_image, dat );
  push_datum_to_port_using_trait( rectified_right_image, dat );
}

// -----------------------------------------------------------------------------
void
measure_objects_process
::_step()
{
  std::vector< kv::object_track_set_sptr > input_tracks;
  std::vector< kv::image_container_sptr > input_images;
  kv::image_container_sptr external_disparity;
  kv::timestamp ts;

  // Check for completion on any input port
  for( auto const& port_name : d->p_port_list )
  {
    if( has_input_port_edge( port_name ) )
    {
      auto port_info = peek_at_port( port_name );

      if( port_info.datum->type() == sprokit::datum::complete )
      {
        _finalize();
        return;
      }
    }
  }

  if( has_input_port_edge_using_trait( timestamp ) )
  {
    auto ts_check = peek_at_port_using_trait( timestamp );

    if( ts_check.datum->type() == sprokit::datum::complete )
    {
      _finalize();
      return;
    }
  }

  // Grab timestamp if connected (declared in make_ports, not in p_port_list)
  if( has_input_port_edge_using_trait( timestamp ) )
  {
    ts = grab_from_port_using_trait( timestamp );
  }

  // Grab optional disparity image if connected
  if( has_input_port_edge_using_trait( disparity_image ) )
  {
    external_disparity = grab_from_port_using_trait( disparity_image );
  }

  // Read dynamic port names (images and track sets)
  for( auto const& port_name : d->p_port_list )
  {
    if( port_name.find( "image" ) != std::string::npos )
    {
      input_images.push_back(
        grab_from_port_as< kv::image_container_sptr >( port_name ) );
    }
    else
    {
      input_tracks.push_back(
        grab_from_port_as< kv::object_track_set_sptr >( port_name ) );
    }
  }

  kv::frame_id_t cur_frame_id = ( ts.has_valid_frame() ?
                                  ts.get_frame() :
                                  d->m_frame_counter );

  d->m_frame_counter++;

  // Invalidate per-frame caches (rectified images, computed disparity,
  // feature matches) so any disparity map computed below is fresh for this
  // frame rather than the previous one.
  d->m_utilities.set_frame_id( cur_frame_id );

  if( input_tracks.size() == 1 )
  {
    // Single track input: create empty 2nd track set so all tracks are
    // treated as left-only and go through the stereo matching path.
    input_tracks.push_back( std::make_shared< kv::object_track_set >() );
  }
  else if( input_tracks.size() != 2 )
  {
    const std::string err = "Expected 1 or 2 track inputs, got "
                            + std::to_string( input_tracks.size() );
    LOG_ERROR( logger(), err );
    throw std::runtime_error( err );
  }

  // Check if images are required but not provided
  bool images_required = false;
  for( const auto& method : d->m_matching_methods )
  {
    if( method_requires_images( method ) )
    {
      images_required = true;
      break;
    }
  }

  if( images_required && input_images.size() < 2 )
  {
    const std::string err = "Input images are required for the specified matching methods "
                            "but were not provided. Please connect image inputs to the process "
                            "or use only 'input_pairs_only' or 'depth_projection' methods.";
    LOG_ERROR( logger(), err );
    throw std::runtime_error( err );
  }

  // Identify all input detections
  typedef std::vector< std::map< kv::track_id_t, kv::detected_object_sptr > > map_t;
  map_t dets( input_tracks.size() );

  for( unsigned i = 0; i < input_tracks.size(); ++i )
  {
    if( !input_tracks[i] )
    {
      continue;
    }

    for( auto& trk : input_tracks[i]->tracks() )
    {
      for( auto& state : *trk )
      {
        auto obj_state =
          std::static_pointer_cast< kwiver::vital::object_track_state >( state );

        if( state->frame() == cur_frame_id )
        {
          dets[i][trk->id()] = obj_state->detection();
        }
      }
    }
  }

  // Identify which detections are matched.
  //
  // Tracker IDs on the left and right are assigned independently, so two
  // unrelated fish can end up sharing an ID purely by coincidence. Auto-
  // matching by ID-equality captures those coincidental pairs before
  // detection_pairing (keypoint_projection) ever runs, and — because it
  // also prevents the legitimate cross-camera fish from ever being
  // considered — leads to permanent misassociation and orphaned true pairs.
  // So: feed every left track into detection_pairing as left-only, every
  // right track as right-only, and let calibrated keypoint projection be
  // the sole authority for per-frame cross-camera matching. Synthetic right
  // tracks (which deliberately reuse left IDs) are created later in _step
  // and aren't in dets[1] yet at this point, so they are unaffected.
  std::vector< kv::track_id_t > common_ids;
  std::vector< kv::track_id_t > left_only_ids;
  for( const auto& itr : dets[0] )
    left_only_ids.push_back( itr.first );

  std::vector< kv::track_id_t > right_only_ids;
  for( const auto& itr : dets[1] )
    right_only_ids.push_back( itr.first );

  // Get camera references
  kv::simple_camera_perspective& left_cam(
    dynamic_cast< kwiver::vital::simple_camera_perspective& >(
      *(d->m_calibration->left())));
  kv::simple_camera_perspective& right_cam(
    dynamic_cast< kwiver::vital::simple_camera_perspective& >(
      *(d->m_calibration->right())));

  // Detection pairing: match left-only and right-only detections.
  //
  // detection_pairing_method may be a comma-separated list (e.g.
  // "disparity_projection,keypoint_projection") for hybrid mode: each
  // method is tried in order on whatever's still unpaired after the
  // previous method ran, so the more selective methods can claim their
  // high-confidence matches first and the broader-recall methods clean
  // up the remainder.
  if( !d->m_settings.detection_pairing_method.empty() &&
      !left_only_ids.empty() && !right_only_ids.empty() )
  {
    auto split_methods = []( const std::string& s )
    {
      std::vector< std::string > out;
      std::string cur;
      for( char c : s )
      {
        if( c == ',' )
        {
          if( !cur.empty() ) out.push_back( cur );
          cur.clear();
        }
        else if( c != ' ' && c != '\t' )
        {
          cur.push_back( c );
        }
      }
      if( !cur.empty() ) out.push_back( cur );
      return out;
    };
    const auto methods = split_methods( d->m_settings.detection_pairing_method );

    feature_matching_algorithms feat_algos;
    feat_algos.feature_detector = d->m_settings.feature_detector;
    feat_algos.descriptor_extractor = d->m_settings.descriptor_extractor;
    feat_algos.feature_matcher = d->m_settings.feature_matcher;

    kv::image_container_sptr img1 = input_images.size() >= 1 ? input_images[0] : nullptr;
    kv::image_container_sptr img2 = input_images.size() >= 2 ? input_images[1] : nullptr;

    for( const auto& method : methods )
    {
      if( left_only_ids.empty() || right_only_ids.empty() )
        break;

      std::vector< kv::detected_object_sptr > left_only_dets, right_only_dets;
      for( const auto& id : left_only_ids )
        left_only_dets.push_back( dets[0][id] );
      for( const auto& id : right_only_ids )
        right_only_dets.push_back( dets[1][id] );

      detection_pairing_params dp;
      dp.method = method;
      dp.threshold = d->m_settings.detection_pairing_threshold;
      dp.default_depth = 0.0;
      dp.require_class_match = d->m_settings.detection_pairing_require_class_match;
      dp.use_optimal_assignment = d->m_settings.detection_pairing_use_optimal_assignment;

      std::vector< std::pair< int, int > > paired;
      if( method == "disparity_projection" )
      {
        auto disparity = d->m_utilities.compute_disparity_for_frame(
          left_cam, right_cam, img1, img2 );

        if( !disparity )
        {
          LOG_WARN( logger(),
            "disparity_projection: no disparity this frame; skipping" );
        }
        else
        {
          auto rectify_left =
            [ & ]( const kv::vector_2d& p ) -> kv::vector_2d
            { return d->m_utilities.rectify_point( p, false ); };
          auto unrectify_right =
            [ & ]( const kv::vector_2d& p ) -> kv::vector_2d
            { return d->m_utilities.unrectify_point( p, true, right_cam ); };

          disparity_projection_matching_options opts;
          opts.max_centroid_distance = dp.threshold;
          opts.require_class_match = dp.require_class_match;
          opts.use_optimal_assignment = dp.use_optimal_assignment;

          paired = find_stereo_matches_disparity_projection(
            left_only_dets, right_only_dets,
            disparity, rectify_left, unrectify_right,
            opts, logger() );
        }
      }
      else
      {
        paired = find_stereo_detection_matches(
          dp, left_only_dets, right_only_dets,
          &left_cam, &right_cam, img1, img2,
          &feat_algos, nullptr, logger() );
      }

      // Merge paired detections from this method's run into common_ids,
      // then prune left_only_ids / right_only_ids so the next method
      // only sees the leftovers.
      std::set< kv::track_id_t > paired_left_ids;
      std::set< kv::track_id_t > paired_right_ids;
      for( const auto& p : paired )
      {
        kv::track_id_t left_id = left_only_ids[ p.first ];
        kv::track_id_t right_id = right_only_ids[ p.second ];

        dets[1][ left_id ] = dets[1][ right_id ];
        if( left_id != right_id )
          dets[1].erase( right_id );

        common_ids.push_back( left_id );
        paired_left_ids.insert( left_id );
        paired_right_ids.insert( right_id );

        if( left_id != right_id )
          d->m_detection_paired_right_ids[ left_id ] = right_id;

        LOG_INFO( logger(),
          "Paired left track " + std::to_string( left_id ) +
          " with right track " + std::to_string( right_id ) +
          " via " + method );
      }

      std::vector< kv::track_id_t > remaining_left;
      for( const auto& id : left_only_ids )
        if( paired_left_ids.find( id ) == paired_left_ids.end() )
          remaining_left.push_back( id );
      left_only_ids = std::move( remaining_left );

      std::vector< kv::track_id_t > remaining_right;
      for( const auto& id : right_only_ids )
        if( paired_right_ids.find( id ) == paired_right_ids.end() )
          remaining_right.push_back( id );
      right_only_ids = std::move( remaining_right );
    }
  }

  // Separate matched detections
  std::vector< kv::track_id_t > fully_matched_ids;
  std::vector< kv::track_id_t > missing_right_kps_ids;

  for( const kv::track_id_t& id : common_ids )
  {
    const auto& det1 = dets[0][id];
    const auto& det2 = dets[1][id];

    if( !det1 || !det2 )
    {
      continue;
    }

    const auto& kp1 = det1->keypoints();
    const auto& kp2 = det2->keypoints();

    bool left_has_kp = ( kp1.find( "head" ) != kp1.end() &&
                         kp1.find( "tail" ) != kp1.end() );
    bool right_has_kp = ( kp2.find( "head" ) != kp2.end() &&
                          kp2.find( "tail" ) != kp2.end() );

    if( left_has_kp && right_has_kp )
    {
      fully_matched_ids.push_back( id );
    }
    else if( left_has_kp && !right_has_kp )
    {
      missing_right_kps_ids.push_back( id );
    }
  }

  // Optional: refine right keypoints of already-paired tracks using a
  // full-image disparity map. The per-camera trackers can place head/tail
  // at slightly different anatomical points; when a stereo_disparity
  // algorithm is configured we can snap each right keypoint to the
  // disparity-implied match of its left counterpart for L/R consistency.
  kv::image_container_sptr refine_disparity;
  if( d->m_settings.refine_keypoints_with_disparity &&
      d->m_settings.stereo_depth_map_algorithm &&
      !fully_matched_ids.empty() &&
      input_images.size() >= 2 &&
      input_images[0] && input_images[1] )
  {
    refine_disparity = d->m_utilities.compute_disparity_for_frame(
      left_cam, right_cam, input_images[0], input_images[1] );

    if( !refine_disparity )
    {
      LOG_WARN( logger(),
        "refine_keypoints_with_disparity: disparity unavailable this "
        "frame; keeping original right keypoints" );
    }
  }

  // Run measurement on fully matched detections
  for( const kv::track_id_t& id : fully_matched_ids )
  {
    const auto& det1 = dets[0][id];
    const auto& det2 = dets[1][id];

    const auto& kp1 = det1->keypoints();
    const auto& kp2 = det2->keypoints();

    kv::vector_2d left_head( kp1.at("head")[0], kp1.at("head")[1] );
    kv::vector_2d right_head( kp2.at("head")[0], kp2.at("head")[1] );
    kv::vector_2d left_tail( kp1.at("tail")[0], kp1.at("tail")[1] );
    kv::vector_2d right_tail( kp2.at("tail")[0], kp2.at("tail")[1] );

    bool head_refined = false, tail_refined = false;
    kv::vector_2d refined_head = right_head;
    kv::vector_2d refined_tail = right_tail;
    if( refine_disparity )
    {
      const int win = d->m_settings.refine_keypoints_disparity_window;
      refined_head = d->m_utilities.refine_right_point_with_disparity(
        refine_disparity, left_head, right_head, right_cam, win,
        &head_refined );
      refined_tail = d->m_utilities.refine_right_point_with_disparity(
        refine_disparity, left_tail, right_tail, right_cam, win,
        &tail_refined );
    }

    // Optional sanity check: reject the track when the tracker-supplied
    // right keypoint disagrees with the disparity-implied position by
    // more than refine_keypoints_max_distance, normalized by the L bbox
    // size. Only keypoints with a valid disparity reading contribute.
    bool inconsistent = false;
    double inconsistent_dist_norm = 0.0;
    if( ( head_refined || tail_refined ) &&
        d->m_settings.refine_keypoints_reject_inconsistent &&
        d->m_settings.refine_keypoints_max_distance > 0.0 )
    {
      const auto& bbox1 = det1->bounding_box();
      double bbox_size = 0.0;
      if( bbox1.is_valid() )
      {
        bbox_size = std::max( bbox1.width(), bbox1.height() );
      }

      if( bbox_size > 0.0 )
      {
        const double thresh = d->m_settings.refine_keypoints_max_distance;
        if( head_refined )
        {
          double d_head = ( refined_head - right_head ).norm() / bbox_size;
          if( d_head > thresh ) { inconsistent = true; }
          inconsistent_dist_norm = std::max( inconsistent_dist_norm, d_head );
        }
        if( tail_refined )
        {
          double d_tail = ( refined_tail - right_tail ).norm() / bbox_size;
          if( d_tail > thresh ) { inconsistent = true; }
          inconsistent_dist_norm = std::max( inconsistent_dist_norm, d_tail );
        }
      }
    }

    if( inconsistent )
    {
      LOG_INFO( logger(), "Track ID " + std::to_string( id ) +
        " disparity inconsistency (max norm dist " +
        std::to_string( inconsistent_dist_norm ) +
        " > threshold " +
        std::to_string( d->m_settings.refine_keypoints_max_distance ) +
        "), skipping measurement" );

      if( d->m_settings.record_stereo_method )
      {
        det1->add_note( ":stereo_method=disparity_inconsistent_rejected" );
        det2->add_note( ":stereo_method=disparity_inconsistent_rejected" );
      }
      continue;
    }

    if( head_refined )
    {
      right_head = refined_head;
      det2->add_keypoint( "head",
        kv::point_2d( right_head.x(), right_head.y() ) );
    }
    if( tail_refined )
    {
      right_tail = refined_tail;
      det2->add_keypoint( "tail",
        kv::point_2d( right_tail.x(), right_tail.y() ) );
    }

    const auto measurement = viame::core::compute_stereo_measurement(
      left_cam, right_cam, left_head, right_head, left_tail, right_tail );

    const std::string method_tag =
      ( head_refined && tail_refined ) ? "input_kps_disparity_refined" :
      ( head_refined || tail_refined ) ? "input_kps_partial_disparity_refined"
                                       : "input_kps_used";

    LOG_INFO( logger(), "Computed Length (" + method_tag + "): " +
              std::to_string( measurement.length ) );
    LOG_INFO( logger(), "  Midpoint (x,y,z): (" + std::to_string( measurement.x ) + ", "
              + std::to_string( measurement.y ) + ", " + std::to_string( measurement.z ) + ")" );
    LOG_INFO( logger(), "  Range: " + std::to_string( measurement.range ) +
              ", RMS: " + std::to_string( measurement.rms ) );

    if( d->m_settings.record_stereo_method )
    {
      det1->add_note( ":stereo_method=" + method_tag );
      det2->add_note( ":stereo_method=" + method_tag );
    }

    add_measurement_attributes( det1, measurement );
    add_measurement_attributes( det2, measurement );
  }

  // Combine left-only IDs and matched IDs missing right keypoints
  std::vector< kv::track_id_t > ids_needing_matching;
  ids_needing_matching.insert( ids_needing_matching.end(),
                               left_only_ids.begin(), left_only_ids.end() );
  ids_needing_matching.insert( ids_needing_matching.end(),
                               missing_right_kps_ids.begin(),
                               missing_right_kps_ids.end() );

  // Check if input_pairs_only is the only method - skip secondary matching
  bool only_input_pairs = ( d->m_matching_methods.size() == 1 &&
                            d->m_matching_methods[0] == "input_pairs_only" );

  if( !ids_needing_matching.empty() && !only_input_pairs )
  {
    // Update frame ID for caching
    d->m_utilities.set_frame_id( cur_frame_id );

    // Get images for methods that need them
    kv::image_container_sptr left_image = input_images.size() >= 1 ? input_images[0] : nullptr;
    kv::image_container_sptr right_image = input_images.size() >= 2 ? input_images[1] : nullptr;

    // Note: DINO crops are now computed per-keypoint inside
    // find_stereo_correspondence for better matching accuracy.

    for( const kv::track_id_t& id : ids_needing_matching )
    {
      const auto& det1 = dets[0][id];

      if( !det1 )
      {
        continue;
      }

      const auto& kp1 = det1->keypoints();

      if( kp1.find( "head" ) == kp1.end() ||
          kp1.find( "tail" ) == kp1.end() )
      {
        LOG_INFO( logger(), "Track ID " + std::to_string( id ) +
                            " missing required keypoints (head/tail)" );
        continue;
      }

      bool is_left_only = ( dets[1].find( id ) == dets[1].end() );

      kv::vector_2d left_head( kp1.at("head")[0], kp1.at("head")[1] );
      kv::vector_2d left_tail( kp1.at("tail")[0], kp1.at("tail")[1] );

      // Check for existing right keypoints
      const kv::vector_2d* right_head_input = nullptr;
      const kv::vector_2d* right_tail_input = nullptr;
      kv::vector_2d right_head_tmp, right_tail_tmp;

      if( !is_left_only )
      {
        const auto& det2 = dets[1].at( id );
        if( det2 )
        {
          const auto& kp2 = det2->keypoints();
          if( kp2.find( "head" ) != kp2.end() && kp2.find( "tail" ) != kp2.end() )
          {
            right_head_tmp = kv::vector_2d( kp2.at("head")[0], kp2.at("head")[1] );
            right_tail_tmp = kv::vector_2d( kp2.at("tail")[0], kp2.at("tail")[1] );
            right_head_input = &right_head_tmp;
            right_tail_input = &right_tail_tmp;
          }
        }
      }

      // Find stereo correspondences using utility function
      auto result = d->m_utilities.find_stereo_correspondence(
        d->m_matching_methods, left_cam, right_cam,
        left_head, left_tail, right_head_input, right_tail_input,
        left_image, right_image, external_disparity );

      if( !result.success )
      {
        LOG_INFO( logger(), "Track ID " + std::to_string( id ) +
                            " no matching method succeeded, skipping" );
        continue;
      }

      // Per-keypoint depth consistency check: if both keypoints matched,
      // triangulate each separately and reject any whose depth is
      // inconsistent with the other. Valid stereo matches on the same
      // object should have similar depths (ratio close to 1.0).
      double max_depth_ratio = d->m_settings.depth_consistency_max_ratio;

      if( max_depth_ratio > 0 && result.head_found && result.tail_found )
      {
        auto head_3d = viame::core::triangulate_point(
          left_cam, right_cam, result.left_head, result.right_head );
        auto tail_3d = viame::core::triangulate_point(
          left_cam, right_cam, result.left_tail, result.right_tail );

        double depth_head = head_3d.z();
        double depth_tail = tail_3d.z();

        if( depth_head > 0 && depth_tail > 0 )
        {
          double depth_ratio = std::max( depth_head, depth_tail ) /
                               std::min( depth_head, depth_tail );

          if( depth_ratio > max_depth_ratio )
          {
            // Keep the keypoint with smaller depth (closer, higher disparity)
            // as that is more likely to be the correct match
            if( depth_head > depth_tail )
            {
              result.head_found = false;
              LOG_INFO( logger(), "Track ID " + std::to_string( id ) +
                " depth check: rejecting head (depth " +
                std::to_string( depth_head ) + " vs tail " +
                std::to_string( depth_tail ) + ", ratio " +
                std::to_string( depth_ratio ) + ")" );
            }
            else
            {
              result.tail_found = false;
              LOG_INFO( logger(), "Track ID " + std::to_string( id ) +
                " depth check: rejecting tail (depth " +
                std::to_string( depth_tail ) + " vs head " +
                std::to_string( depth_head ) + ", ratio " +
                std::to_string( depth_ratio ) + ")" );
            }
            result.success = result.head_found || result.tail_found;

            if( !result.success )
            {
              LOG_INFO( logger(), "Track ID " + std::to_string( id ) +
                " depth check: both keypoints rejected, skipping" );
              continue;
            }
          }
        }
      }

      bool both_found = result.head_found && result.tail_found;

      if( both_found )
      {
        // Compute full measurement (requires both keypoints)
        const auto measurement = viame::core::compute_stereo_measurement(
          left_cam, right_cam,
          result.left_head, result.right_head,
          result.left_tail, result.right_tail );

        LOG_INFO( logger(), "Computed Length (" + result.method_used + "): " +
                            std::to_string( measurement.length ) );
        LOG_INFO( logger(), "  Midpoint (x,y,z): (" + std::to_string( measurement.x ) + ", "
                  + std::to_string( measurement.y ) + ", " + std::to_string( measurement.z ) + ")" );
        LOG_INFO( logger(), "  Range: " + std::to_string( measurement.range ) +
                  ", RMS: " + std::to_string( measurement.rms ) );

        if( d->m_settings.record_stereo_method )
        {
          det1->add_note( ":stereo_method=" + result.method_used );
        }

        add_measurement_attributes( det1, measurement );
      }
      else
      {
        // Partial match: record method but no length measurement
        std::string matched_kp = result.head_found ? "head" : "tail";
        LOG_INFO( logger(), "Track ID " + std::to_string( id ) +
                            " partial match (" + result.method_used + "): only " +
                            matched_kp + " found, no length measurement" );

        if( d->m_settings.record_stereo_method )
        {
          det1->add_note( ":stereo_method=" + result.method_used + "_partial" );
        }
      }

      if( is_left_only )
      {
        // Create right detection with available keypoints
        kv::vector_2d bbox_pt1, bbox_pt2;

        if( both_found )
        {
          bbox_pt1 = result.right_head;
          bbox_pt2 = result.right_tail;
        }
        else if( result.head_found )
        {
          // Single point: create small bbox around matched keypoint
          bbox_pt1 = kv::vector_2d( result.right_head.x() - 10, result.right_head.y() - 10 );
          bbox_pt2 = kv::vector_2d( result.right_head.x() + 10, result.right_head.y() + 10 );
        }
        else
        {
          bbox_pt1 = kv::vector_2d( result.right_tail.x() - 10, result.right_tail.y() - 10 );
          bbox_pt2 = kv::vector_2d( result.right_tail.x() + 10, result.right_tail.y() + 10 );
        }

        kv::bounding_box_d right_bbox = both_found
          ? d->m_utilities.compute_bbox_from_keypoints( bbox_pt1, bbox_pt2 )
          : kv::bounding_box_d( bbox_pt1.x(), bbox_pt1.y(), bbox_pt2.x(), bbox_pt2.y() );

        auto det2 = std::make_shared< kv::detected_object >( right_bbox );

        if( result.head_found )
          det2->add_keypoint( "head", kv::point_2d( result.right_head.x(), result.right_head.y() ) );
        if( result.tail_found )
          det2->add_keypoint( "tail", kv::point_2d( result.right_tail.x(), result.right_tail.y() ) );

        if( det1->type() )
        {
          det2->set_type( det1->type() );
        }
        det2->set_confidence( det1->confidence() );

        if( d->m_settings.record_stereo_method )
        {
          det2->add_note( ":stereo_method=" + result.method_used +
                          ( both_found ? "" : "_partial" ) );
        }

        if( both_found )
        {
          const auto measurement = viame::core::compute_stereo_measurement(
            left_cam, right_cam,
            result.left_head, result.right_head,
            result.left_tail, result.right_tail );
          add_measurement_attributes( det2, measurement );
        }

        // Look up or create a persistent right track for this left-only ID
        kv::track_sptr right_track;
        auto it = d->m_created_right_tracks.find( id );

        if( it != d->m_created_right_tracks.end() )
        {
          right_track = it->second;
        }
        else
        {
          right_track = kv::track::create();
          right_track->set_id( id );
          d->m_created_right_tracks[id] = right_track;
        }

        kv::time_usec_t time_usec = ts.has_valid_time() ? ts.get_time_usec() : 0;
        auto new_state = std::make_shared< kv::object_track_state >(
          cur_frame_id, time_usec, det2 );
        right_track->append( std::move( new_state ) );

        LOG_INFO( logger(), "Created right detection for track ID " + std::to_string( id ) +
                  ( both_found ? "" : " (partial)" ) );
      }
      else
      {
        auto& det2 = dets[1][id];

        if( result.head_found )
          det2->add_keypoint( "head", kv::point_2d( result.right_head.x(), result.right_head.y() ) );
        if( result.tail_found )
          det2->add_keypoint( "tail", kv::point_2d( result.right_tail.x(), result.right_tail.y() ) );

        if( d->m_settings.record_stereo_method )
        {
          det2->add_note( ":stereo_method=" + result.method_used +
                          ( both_found ? "" : "_partial" ) );
        }

        if( both_found )
        {
          const auto measurement = viame::core::compute_stereo_measurement(
            left_cam, right_cam,
            result.left_head, result.right_head,
            result.left_tail, result.right_tail );
          add_measurement_attributes( det2, measurement );
        }
      }
    }

    // Per-keypoint DINO crops are now computed and discarded inline,
    // no per-frame state to clear.
  }

  // Ensure output track sets exist
  if( !input_tracks[0] )
  {
    input_tracks[0] = std::make_shared< kv::object_track_set >();
  }
  if( !input_tracks[1] )
  {
    input_tracks[1] = std::make_shared< kv::object_track_set >();
  }

  // Merge persistent right tracks (created for left-only detections) into output
  for( const auto& pair : d->m_created_right_tracks )
  {
    if( !input_tracks[1]->get_track( pair.first ) )
    {
      input_tracks[1]->insert( pair.second );
    }
  }

  if( d->m_track_pairer.accumulation_enabled() )
  {
    // Accumulation mode: feed current-frame detections + pairings into the
    // shared accumulator. Final unified tracks are emitted in _finalize()
    // once the whole stream has been seen — this avoids the per-frame
    // output-ID reassignment that causes downstream writers to accumulate
    // stale duplicate track entries.
    std::vector< kv::detected_object_sptr > dets1_vec, dets2_vec;
    std::vector< kv::track_id_t > tids1, tids2;
    std::map< kv::track_id_t, int > tid1_to_idx, tid2_to_idx;

    for( const auto& trk : input_tracks[0]->tracks() )
    {
      auto it = trk->find( cur_frame_id );
      if( it == trk->end() ) continue;
      auto ots = std::dynamic_pointer_cast< kv::object_track_state >( *it );
      if( !ots || !ots->detection() ) continue;
      tid1_to_idx[ trk->id() ] = static_cast< int >( dets1_vec.size() );
      tids1.push_back( trk->id() );
      dets1_vec.push_back( ots->detection() );
    }
    for( const auto& trk : input_tracks[1]->tracks() )
    {
      auto it = trk->find( cur_frame_id );
      if( it == trk->end() ) continue;
      auto ots = std::dynamic_pointer_cast< kv::object_track_state >( *it );
      if( !ots || !ots->detection() ) continue;
      tid2_to_idx[ trk->id() ] = static_cast< int >( dets2_vec.size() );
      tids2.push_back( trk->id() );
      dets2_vec.push_back( ots->detection() );
    }

    // Reject per-frame pairings whose geometry is inconsistent, before they
    // enter the union-find. Three independent calibration-informed filters:
    //   - max_stereo_rms: triangulation reprojection error
    //   - max_bbox_y_center_offset: epipolar residual proxy for near-rectified
    //     rigs — catches pairs at different image heights (unrelated fish)
    //     even when rms happens to be low
    //   - max_bbox_area_ratio: catches same-tracker-ID false matches between
    //     targets of very different image size (small fish vs school, etc.)
    auto pair_ok = [&]( int left_idx, int right_idx ) -> bool
    {
      const auto& ldet = dets1_vec[ left_idx ];
      const auto& rdet = dets2_vec[ right_idx ];

      if( d->m_max_stereo_rms > 0.0 )
      {
        double rms = parse_stereo_rms_from_notes( ldet );
        if( rms >= 0.0 && rms > d->m_max_stereo_rms )
          return false;
      }

      if( d->m_max_bbox_y_center_offset > 0.0 && ldet && rdet )
      {
        const auto& lbb = ldet->bounding_box();
        const auto& rbb = rdet->bounding_box();
        double lcy = 0.5 * ( lbb.min_y() + lbb.max_y() );
        double rcy = 0.5 * ( rbb.min_y() + rbb.max_y() );
        if( std::abs( lcy - rcy ) > d->m_max_bbox_y_center_offset )
          return false;
      }

      if( d->m_max_bbox_area_ratio > 0.0 && ldet && rdet )
      {
        const auto& lbb = ldet->bounding_box();
        const auto& rbb = rdet->bounding_box();
        double la = std::max( 1.0, lbb.width() * lbb.height() );
        double ra = std::max( 1.0, rbb.width() * rbb.height() );
        double ratio = ( la > ra ) ? ( la / ra ) : ( ra / la );
        if( ratio > d->m_max_bbox_area_ratio )
          return false;
      }

      return true;
    };

    std::vector< std::pair< int, int > > match_pairs;
    for( const auto& entry : tid1_to_idx )
    {
      auto it2 = tid2_to_idx.find( entry.first );
      if( it2 != tid2_to_idx.end() && pair_ok( entry.second, it2->second ) )
        match_pairs.push_back( { entry.second, it2->second } );
    }
    for( const auto& dp : d->m_detection_paired_right_ids )
    {
      auto it1 = tid1_to_idx.find( dp.first );
      auto it2 = tid2_to_idx.find( dp.second );
      if( it1 != tid1_to_idx.end() && it2 != tid2_to_idx.end() &&
          pair_ok( it1->second, it2->second ) )
        match_pairs.push_back( { it1->second, it2->second } );
    }

    d->m_track_pairer.accumulate_frame_pairings(
      match_pairs, dets1_vec, dets2_vec, tids1, tids2, ts );

    // Push empty track sets per frame — real object_track_set values so
    // downstream writers can cast them (empty_datum() would trip their
    // type-cast). The resolved unified track sets are emitted in
    // _finalize() once the whole stream has been accumulated.
    auto empty_set = std::make_shared< kv::object_track_set >();
    push_to_port_using_trait( object_track_set1, empty_set );
    push_to_port_using_trait( object_track_set2, empty_set );
  }
  else
  {
    // Per-frame mode: union-find remap + class averaging each frame.
    // Warning: output track IDs can shift when a late pairing arrives,
    // which can cause downstream writers that key by track ID to retain
    // stale copies. Prefer accumulation mode for batch CSV outputs.
    if( input_tracks[0] && input_tracks[1] &&
        input_tracks[0]->size() > 0 && input_tracks[1]->size() > 0 )
    {
      // Build match list from common IDs (tracks with same ID in both sets)
      std::vector< kv::track_id_t > tids1, tids2;
      std::vector< std::pair< int, int > > match_pairs;

      for( const auto& trk : input_tracks[0]->tracks() )
      {
        auto it = trk->find( cur_frame_id );
        if( it != trk->end() )
        {
          auto ots = std::dynamic_pointer_cast< kv::object_track_state >( *it );
          if( ots && ots->detection() )
            tids1.push_back( trk->id() );
        }
      }

      for( const auto& trk : input_tracks[1]->tracks() )
      {
        auto it = trk->find( cur_frame_id );
        if( it != trk->end() )
        {
          auto ots = std::dynamic_pointer_cast< kv::object_track_state >( *it );
          if( ots && ots->detection() )
            tids2.push_back( trk->id() );
        }
      }

      std::map< kv::track_id_t, int > left_idx, right_idx;
      for( int i = 0; i < static_cast< int >( tids1.size() ); ++i )
        left_idx[tids1[i]] = i;
      for( int i = 0; i < static_cast< int >( tids2.size() ); ++i )
        right_idx[tids2[i]] = i;

      for( int i = 0; i < static_cast< int >( tids1.size() ); ++i )
      {
        auto rit = right_idx.find( tids1[i] );
        if( rit != right_idx.end() )
          match_pairs.push_back( { i, rit->second } );
      }

      for( const auto& dp : d->m_detection_paired_right_ids )
      {
        auto lit = left_idx.find( dp.first );
        auto rit = right_idx.find( dp.second );
        if( lit != left_idx.end() && rit != right_idx.end() )
          match_pairs.push_back( { lit->second, rit->second } );
      }

      std::vector< kv::track_sptr > out1, out2;
      d->m_track_pairer.remap_tracks_per_frame(
        input_tracks[0], input_tracks[1], match_pairs,
        tids1, tids2, out1, out2 );

      input_tracks[0] = std::make_shared< kv::object_track_set >( out1 );
      input_tracks[1] = std::make_shared< kv::object_track_set >( out2 );
    }

    if( d->m_min_track_states > 0 )
    {
      std::map< kv::track_id_t, unsigned > state_counts;
      for( unsigned i = 0; i < 2; ++i )
      {
        if( !input_tracks[i] )
          continue;
        for( const auto& trk : input_tracks[i]->tracks() )
        {
          state_counts[trk->id()] +=
            static_cast< unsigned >( trk->size() );
        }
      }

      for( unsigned i = 0; i < 2; ++i )
      {
        if( !input_tracks[i] )
          continue;
        std::vector< kv::track_sptr > kept;
        for( const auto& trk : input_tracks[i]->tracks() )
        {
          auto it = state_counts.find( trk->id() );
          if( it != state_counts.end() &&
              it->second >= d->m_min_track_states )
          {
            kept.push_back( trk );
          }
        }
        input_tracks[i] = std::make_shared< kv::object_track_set >( kept );
      }
    }

    push_to_port_using_trait( object_track_set1, input_tracks[0] );
    push_to_port_using_trait( object_track_set2, input_tracks[1] );
  }

  push_to_port_using_trait( timestamp, ts );

  // Push disparity image if computed
  kv::image_container_sptr computed_disparity = d->m_utilities.get_cached_disparity();
  push_to_port_using_trait( disparity_image, computed_disparity );

  // Push rectified images if available
  kv::image_container_sptr rectified_left = d->m_utilities.get_cached_rectified_left();
  kv::image_container_sptr rectified_right = d->m_utilities.get_cached_rectified_right();
  push_to_port_using_trait( rectified_left_image, rectified_left );
  push_to_port_using_trait( rectified_right_image, rectified_right );
}

} // end namespace core

} // end namespace viame
