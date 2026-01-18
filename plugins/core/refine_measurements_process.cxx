/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Refine measurements in object detections via multiple methods
 */

#include "refine_measurements_process.h"

#include <vital/vital_types.h>
#include <vital/types/image_container.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/object_track_set.h>
#include <vital/types/metadata.h>
#include <vital/types/metadata_traits.h>

#include <sprokit/processes/kwiver_type_traits.h>

#include <Eigen/Core>

#include <cmath>


namespace kv = kwiver::vital;

namespace viame
{

namespace core
{

create_config_trait( recompute_all, bool, "false",
  "If set, recompute lengths for all detections using GSD, even those "
  "already containing lengths." );
create_config_trait( output_multiple, bool, "false",
  "Allow outputting multiple possible lengths for each detection"  );
create_config_trait( output_conf_level, bool, "false",
  "Output length confidence metric"  );
create_config_trait( min_valid, double, "-1.0",
  "Minimum allowed valid measurement"  );
create_config_trait( max_valid, double, "-1.0",
  "Maximum allowed valid measurement"  );
create_config_trait( history_length, unsigned, "0",
  "History to consider when averaging GSDs" );
create_config_trait( exp_factor, double, "-1.0",
  "Exponential averaging factor to consider when averaging" );
create_config_trait( border_factor, unsigned, "0",
  "Treat detections this many pixels near image border as ambiguous" );
create_config_trait( percentile, double, "0.45",
  "Percentile GSD to use when combining multiple estimates" );
create_config_trait( intrinsics, std::string, "1 0 0 0 1 0 0 0 1",
  "Camera calibration for use with metadata" );

typedef std::pair< double, double > point_t;

// =============================================================================
// Private implementation class
class refine_measurements_process::priv
{
public:
  explicit priv( refine_measurements_process* parent );
  ~priv();

  // Configuration settings
  bool m_recompute_all;
  bool m_output_multiple;
  bool m_output_conf_level;
  double m_min_valid;
  double m_max_valid;
  unsigned m_history_length;
  double m_exp_factor;
  unsigned m_border_factor;
  double m_percentile;
  Eigen::Matrix3d m_intrinsics;

  // Internal variables
  double m_last_gsd;
  std::vector< double > m_history;

  // Other variables
  refine_measurements_process* parent;

  // Helper functions
  double percentile( std::vector< double >& vec  );
  bool is_border( const kv::bounding_box_d& box, unsigned w, unsigned h );
};


// -----------------------------------------------------------------------------
refine_measurements_process::priv
::priv( refine_measurements_process* ptr )
  : m_recompute_all( false )
  , m_output_multiple( false )
  , m_output_conf_level( false )
  , m_min_valid( -1.0 )
  , m_max_valid( -1.0 )
  , m_history_length( 0 )
  , m_exp_factor( -1.0 )
  , m_border_factor( 0 )
  , m_percentile( 0.45 )
  , m_intrinsics()
  , m_last_gsd( -1.0 )
  , m_history()
  , parent( ptr )
{
}


refine_measurements_process::priv
::~priv()
{
}


double
refine_measurements_process::priv
::percentile( std::vector< double >& vec )
{
  if( vec.empty() || m_percentile >= 1.0 )
  {
    return -1.0;
  }

  std::sort( vec.begin(), vec.end() );
  unsigned ind = static_cast< unsigned >( m_percentile * vec.size() );
  return vec[ ind ];
}

bool
refine_measurements_process::priv
::is_border( const kv::bounding_box_d& box, unsigned w, unsigned h )
{
  return ( box.min_x() <= m_border_factor ||
           box.min_y() <= m_border_factor ||
           box.max_x() >= w - m_border_factor ||
           box.max_y() >= h - m_border_factor );
}

point_t
compute_ground_position( const point_t& pos, const Eigen::MatrixXd& inv )
{
  Eigen::Vector3d unadj = inv * Eigen::Vector3d( pos.first, pos.second, 1.0 );  

  if( unadj( 2 ) > 0 )
  {
    return point_t( unadj( 0 ) / unadj( 2 ), unadj( 1 ) / unadj( 2 ) );
  }

  return point_t( 0.0, 0.0 );
}

void
remove_column( Eigen::MatrixXd& matrix, unsigned int col )
{
  unsigned int rows = matrix.rows();
  unsigned int cols = matrix.cols()-1;

  if( col < cols )
  {
    matrix.block( 0, col, rows, cols-col ) = matrix.block( 0, col+1, rows, cols-col );
  }

  matrix.conservativeResize( rows, cols );
}


// =============================================================================
refine_measurements_process
::refine_measurements_process( kv::config_block_sptr const& config )
  : process( config ),
    d( new refine_measurements_process::priv( this ) )
{
  make_ports();
  make_config();

  set_data_checking_level( check_valid );
}


refine_measurements_process
::~refine_measurements_process()
{
}


// -----------------------------------------------------------------------------
void
refine_measurements_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- inputs --
  declare_input_port_using_trait( image, optional );
  declare_input_port_using_trait( timestamp, optional );
  declare_input_port_using_trait( detected_object_set, optional );
  declare_input_port_using_trait( object_track_set, optional );
  declare_input_port_using_trait( gsd, optional );
  declare_input_port_using_trait( metadata, optional );

  // -- outputs --
  declare_output_port_using_trait( timestamp, optional );
  declare_output_port_using_trait( detected_object_set, optional );
  declare_output_port_using_trait( object_track_set, optional );
}

// -----------------------------------------------------------------------------
void
refine_measurements_process
::make_config()
{
  declare_config_using_trait( recompute_all );
  declare_config_using_trait( output_multiple );
  declare_config_using_trait( output_conf_level );
  declare_config_using_trait( min_valid );
  declare_config_using_trait( max_valid );
  declare_config_using_trait( history_length );
  declare_config_using_trait( exp_factor );
  declare_config_using_trait( border_factor );
  declare_config_using_trait( percentile );
  declare_config_using_trait( intrinsics );
}

// -----------------------------------------------------------------------------
void
refine_measurements_process
::_configure()
{
  d->m_recompute_all = config_value_using_trait( recompute_all );
  d->m_output_multiple = config_value_using_trait( output_multiple );
  d->m_output_conf_level = config_value_using_trait( output_conf_level );
  d->m_min_valid = config_value_using_trait( min_valid );
  d->m_max_valid = config_value_using_trait( max_valid );
  d->m_history_length = config_value_using_trait( history_length );
  d->m_exp_factor = config_value_using_trait( exp_factor );
  d->m_border_factor = config_value_using_trait( border_factor );
  d->m_percentile = config_value_using_trait( percentile );

  std::string intrinsics_str = config_value_using_trait( intrinsics );

  double a11, a12, a13, a21, a22, a23, a31, a32, a33;
  std::istringstream ss( intrinsics_str );
  ss >> a11 >> a12 >> a13 >> a21 >> a22 >> a23 >> a31 >> a32 >> a33;
  d->m_intrinsics << a11, a12, a13, a21, a22, a23, a31, a32, a33;
}

// -----------------------------------------------------------------------------
void
refine_measurements_process
::_step()
{
  kv::object_track_set_sptr input_tracks;
  kv::detected_object_set_sptr input_dets;
  kv::image_container_sptr image;
  kv::timestamp timestamp;
  kv::metadata_vector metadata;
  double external_gsd = -1.0;

  auto port_info = peek_at_port_using_trait( detected_object_set );

  if( port_info.datum->type() == sprokit::datum::complete )
  {
    mark_process_as_complete();

    const sprokit::datum_t dat = sprokit::datum::complete_datum();

    push_datum_to_port_using_trait( detected_object_set, dat );
    push_datum_to_port_using_trait( object_track_set, dat );
    return;
  }

  if( has_input_port_edge_using_trait( detected_object_set ) )
  {
    input_dets = grab_from_port_using_trait( detected_object_set );
  }
  if( has_input_port_edge_using_trait( object_track_set ) )
  {
    input_tracks = grab_from_port_using_trait( object_track_set );
  }
  if( has_input_port_edge_using_trait( timestamp ) )
  {
    timestamp = grab_from_port_using_trait( timestamp );
  }
  if( has_input_port_edge_using_trait( image ) )
  {
    image = grab_from_port_using_trait( image );
  }
  if( has_input_port_edge_using_trait( gsd ) )
  {
    external_gsd = grab_from_port_using_trait( gsd );
  }
  if( has_input_port_edge_using_trait( metadata ) )
  {
    metadata = grab_from_port_using_trait( metadata );
  }

  const unsigned detection_count = ( input_dets ? input_dets->size() : 0 );

  const unsigned img_height = ( image ? image->height() : 0 );
  const unsigned img_width = ( image ? image->width() : 0 );

  std::vector< unsigned > length_conf( detection_count, 0 );
  std::vector< double > lengths( detection_count, -1.0 );

  const std::string conf_str[5] = { "none", "very_low", "low", "medium", "high" };
  std::vector< double > conf_ests[5];
  unsigned highest_conf = 0;

  if( input_dets )
  {
    unsigned ind = 0;

    for( auto det : *input_dets )
    {
      if( !det->notes().empty() && det->bounding_box().width() > 0 )
      {
        for( auto note : det->notes() )
        {
          if( note.size() > 8 && note.substr( 0, 8 ) == ":length=" )
          {
            double lth = std::stod( note.substr( 8 ) );
            double est = lth / det->bounding_box().width();

            lengths[ ind ] = lth;

            if( ( d->m_min_valid > 0.0 && lth < d->m_min_valid ) ||
                ( d->m_max_valid > 0.0 && lth > d->m_max_valid ) )
            {

              length_conf[ ind ] = 1;
              highest_conf = std::max( highest_conf, 1u );
              conf_ests[1].push_back( est );
            }
            else if( d->is_border( det->bounding_box(), img_width, img_height ) )
            {
              length_conf[ ind ] = 2;
              highest_conf = std::max( highest_conf, 2u );
              conf_ests[2].push_back( est );
            }
            else
            {
              length_conf[ ind ] = 3;
              highest_conf = std::max( highest_conf, 3u );
              conf_ests[3].push_back( est );
            }
          }
        }
      }

      ind++;
    }
  }

  double initial_gsd_est = -1.0;

  if( external_gsd > 0.0 )
  {
    initial_gsd_est = external_gsd;
    d->m_last_gsd = initial_gsd_est;
  }
  else if( highest_conf > 1 )
  {
    initial_gsd_est = d->percentile( conf_ests[ highest_conf ] );
    d->m_last_gsd = initial_gsd_est;
  }
  else if( d->m_last_gsd > 0 )
  {
    initial_gsd_est = d->m_last_gsd;
  }

  double gsd_to_use = initial_gsd_est;

  if( input_dets && gsd_to_use > 0.0 )
  {
    unsigned ind = 0;

    for( auto det : *input_dets )
    {
      unsigned conf_cat = 0;

      if( ( d->m_recompute_all ||
            det->notes().empty() ||
            length_conf[ ind ] <= 2 ) &&
          det->bounding_box().width() > 0 )
      {
        if( !d->m_output_multiple )
        {
          det->clear_notes();
        }

        double lth = det->bounding_box().width() * gsd_to_use;

        if( ( d->m_min_valid <= 0.0 || lth >= d->m_min_valid ) &&
            ( d->m_max_valid <= 0.0 || lth <= d->m_max_valid ) )
        {
          det->set_length( lth );
          conf_cat = ( length_conf[ ind ] ? length_conf[ ind ] : 3 );
        }
      }
      else
      {
        if( det->bounding_box().width() > 0 )
        {
          double l1 = det->bounding_box().width() * gsd_to_use;
          double l2 = lengths[ ind ];

          if( std::abs( ( l1 - l2 ) / l2 ) < 0.10 )
          {
            conf_cat = 4;
          }
          else
          {
            conf_cat = length_conf[ ind ];
          }
        }
        else
        {
          conf_cat = length_conf[ ind ];
        }
      }

      if( d->m_output_conf_level )
      {
        det->add_note( ":length_conf=" + conf_str[ conf_cat ] );
      }

      ind++;
    }
  }

#define CHECK_FIELD( VAR, METAID )                                 \
  {                                                                \
    auto md_ret = md->find( METAID );                              \
                                                                   \
    if( md_ret.is_valid() )                                        \
    {                                                              \
      VAR = md_ret.as_double();                                    \
      has_metadata = true;                                         \
    }                                                              \
  }

  if( !metadata.empty() && input_dets )
  {
    double yaw = 0.0, pitch = 0.0, roll = 0.0, alt = 0.0;
    bool has_metadata = false;

    for( auto md : metadata )
    {
      CHECK_FIELD( yaw, kwiver::vital::VITAL_META_DENSITY_ALTITUDE );
      CHECK_FIELD( pitch, kwiver::vital::VITAL_META_DENSITY_ALTITUDE );
      CHECK_FIELD( roll, kwiver::vital::VITAL_META_DENSITY_ALTITUDE );
      CHECK_FIELD( alt, kwiver::vital::VITAL_META_DENSITY_ALTITUDE );
    }

    if( !has_metadata )
    {
      goto output_objs;
    }

    Eigen::AngleAxisd roll_angle( roll * 0.017453, Eigen::Vector3d::UnitZ() );
    Eigen::AngleAxisd yaw_angle( yaw * 0.017453, Eigen::Vector3d::UnitY() );
    Eigen::AngleAxisd pitch_angle( pitch * 0.017453, Eigen::Vector3d::UnitX() );

    Eigen::Quaternion<double> q = roll_angle * yaw_angle * pitch_angle;

    Eigen::Matrix3d rotation_matrix = q.matrix();
    Eigen::Vector3d translation_matrix( 0, 0, alt * 1000 );

    Eigen::MatrixXd perspective( 3, 4 );
    perspective << rotation_matrix, translation_matrix;

    Eigen::MatrixXd camera = d->m_intrinsics * perspective;
    remove_column( camera, 2 );

    Eigen::MatrixXd inverse = camera.inverse();

    for( auto det : *input_dets )
    {
      if( ( d->m_recompute_all ||
            det->notes().empty() ) &&
          det->bounding_box().width() > 0 )
      {
        if( !d->m_output_multiple )
        {
          det->clear_notes();
        }

        double img_y1 = ( det->bounding_box().min_y() + det->bounding_box().max_y() ) / 2;

        double img_x1 = det->bounding_box().min_x();
        double img_x2 = det->bounding_box().max_x();
 
        point_t world_xy1 = compute_ground_position( point_t( img_x1, img_y1 ), inverse );
        point_t world_xy2 = compute_ground_position( point_t( img_x2, img_y1 ), inverse );

        double lth = std::sqrt( std::pow( world_xy1.first - world_xy2.first, 2 ) +
                                std::pow( world_xy1.second - world_xy2.second, 2 ) );

        if( ( d->m_min_valid <= 0.0 || lth >= d->m_min_valid ) &&
            ( d->m_max_valid <= 0.0 || lth <= d->m_max_valid ) )
        {
          det->set_length( lth );
        }
      }
    }
  }

  output_objs:
  push_to_port_using_trait( detected_object_set, input_dets );
  push_to_port_using_trait( timestamp, timestamp );
}

} // end namespace core

} // end namespace viame
