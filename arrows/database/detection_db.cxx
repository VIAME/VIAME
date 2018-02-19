
/*ckwg +5
 * Copyright 2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "detection_db.h"
#include "database/connection/postgresql_db_connection.h"

#include <vnl/io/vnl_io_matrix.h>
#include <tracking_data/vsl/vil_image_resource_sptr_io.h>

#include <logger/logger.h>
VIDTK_LOGGER("detection_db_cxx");

namespace vidtk
{

detection_db
::detection_db( db_connection_params const& params )
  : connect_string_( "" )
{
  connect_string_ += ( params.db_type + ":" );
#ifdef MODULE_PATH
  connect_string_ += "@modules_path="  MODULE_PATH;
#endif

  if (params.db_type == "postgresql")
  {
    // @todo: Make this construct into generically located macro
    connect_string_ += ( ";@blob=bytea");
    connect_string_ += ( ";@pool_size=10");
    connect_string_ += ( ";host=" + params.db_host);
    connect_string_ += ( ";user=" + params.db_user);
    connect_string_ += ( ";password=" + params.db_pass);
    connect_string_ += ( ";dbname=" + params.db_name);
    connect_string_ += ( ";port=" + params.db_port);
    //  db_conn_pool_ = cppdb::pool::create( connect_string_ );
  }
  else if (params.db_type == "sqlite3")
  {
    connect_string_ += ( ";db=" + params.db_name );
  }
}

detection_db
::~detection_db()
{

}

vxl_int_32
detection_db
::add_weighted_image(weighted_image const & w_img)
{
  OPEN_DB_CONN;

  //  cppdb::session sql( db_conn_pool_->open() );
  int col_index = 1;
  std::stringstream query;
  query << "insert into weighted_image ";
  query << "(wi_image, wi_session_id, wi_uuid, wi_frame_number, wi_frame_time, wi_upper_left, ";
  query << "wi_upper_right, wi_lower_right, wi_lower_left ) values(?,?,?,?,?,";
  query << SPATIAL_LON_LAT_POINT_INSERT_STMT << ",";
  query << SPATIAL_LON_LAT_POINT_INSERT_STMT << ",";
  query << SPATIAL_LON_LAT_POINT_INSERT_STMT << ",";
  query << SPATIAL_LON_LAT_POINT_INSERT_STMT << ")";

  cppdb::transaction guard(db_conn_);
  vxl_int_32 wi_id = 0;

  try
  {
    cppdb::statement stmt = db_conn_.prepare( query.str() );

    std::stringstream img_stream;
    vnl_matrix<float> M = w_img.image;
    vsl_b_ostream img_bss(&img_stream);
    vsl_b_write(img_bss, M);
    img_stream.flush();
    std::stringstream bind_stream(img_stream.str());

    // bind image
    stmt.bind( col_index++, bind_stream);

    stmt.bind( col_index++, this->session_id_ );

    std::stringstream uuid_strstrm;
    uuid_strstrm << w_img.wi_uuid;
    std::string usrt = uuid_strstrm.str();
    stmt.bind( col_index++, uuid_strstrm.str() );

    unsigned fn = 0;
    double ft = 0;
    if (w_img.time.is_valid())
    {
      fn = w_img.time.frame_number();
      ft = w_img.time.time();
    }
    stmt.bind( col_index++, fn );
    stmt.bind( col_index++, ft );

    geo_coord::geo_bounds bns = w_img.bounds;
    geo_coord::geo_coordinate ul = bns.get_upper_left();
    stmt.bind( col_index++, ul.get_longitude() );
    stmt.bind( col_index++, ul.get_latitude() );

    geo_coord::geo_coordinate ur = bns.get_upper_right();
    stmt.bind( col_index++, ur.get_longitude() );
    stmt.bind( col_index++, ur.get_latitude() );

    geo_coord::geo_coordinate lr = bns.get_lower_right();
    stmt.bind( col_index++, lr.get_longitude() );
    stmt.bind( col_index++, lr.get_latitude() );

    geo_coord::geo_coordinate ll = bns.get_lower_left();
    stmt.bind( col_index++, ll.get_longitude() );
    stmt.bind( col_index++, ll.get_latitude() );

    stmt.exec();
    wi_id = stmt.sequence_last( "weighted_image_wi_id_seq" );
    guard.commit();
  }
  catch(cppdb::cppdb_error const &e)
  {
    guard.rollback();
    LOG_ERROR( e.what() );
    CLOSE_DB_CONN;
    return 0;
  }

  CLOSE_DB_CONN;
  return wi_id;
}


bool
detection_db
::get_all_weighted_images( std::vector<weighted_image>& w_img_list )
{
  OPEN_DB_CONN;

  std::stringstream query;
  query << "select wi_image, wi_uuid, wi_frame_number, wi_frame_time, ";

  query << "st_y(wi_upper_left) as ul_lat, ";
  query << "st_x(wi_upper_left) as ul_lon, ";

  query << "st_y(wi_upper_right) as ur_lat, ";
  query << "st_x(wi_upper_right) as ur_lon, ";

  query << "st_y(wi_lower_right) as lr_lat, ";
  query << "st_x(wi_lower_right) as lr_lon, ";

  query << "st_y(wi_lower_left) as ll_lat, ";
  query << "st_x(wi_lower_left) as ll_lon ";

  query << "from weighted_image where wi_session_id = ?";

  try
  {
    cppdb::statement stmt = db_conn_.prepare( query.str() );
    stmt.bind( 1, this->session_id_ );

    std::vector<float> img;
    cppdb::result rs = stmt.query();

    // There is only 1 entry with a given ID
    while (rs.next())
    {
      weighted_image w_img;
      int col_index = 0;
      std::stringstream img_stream;

      rs.fetch( col_index++, img_stream);
      if (img_stream)
      {
        img_stream.peek();
        if( img_stream.good() )
        {
          vsl_b_istream biss(&img_stream);
          if (img_stream.good())
          {
            vnl_matrix<float> M;
            vsl_b_read(biss, M);
            w_img.image = M;
          }
        }
      }

      std::string uuid_str;
      rs.fetch( col_index++, uuid_str );
      std::stringstream uuid_strstrm ( uuid_str );
      uuid_strstrm >> w_img.wi_uuid;

      unsigned frame_number;
      rs.fetch( col_index++, frame_number );

      double frame_time;
      rs.fetch( col_index++, frame_time );
      timestamp ts( frame_time, frame_number );
      w_img.time = ts;

      double wi_upper_left_lat;
      rs.fetch( col_index++, wi_upper_left_lat );
      double wi_upper_left_lon;
      rs.fetch( col_index++, wi_upper_left_lon );
      geo_coord::geo_coordinate ul( wi_upper_left_lat, wi_upper_left_lon );
      w_img.bounds.add( ul);

      double wi_upper_right_lat;
      rs.fetch( col_index++, wi_upper_right_lat );
      double wi_upper_right_lon;
      rs.fetch( col_index++, wi_upper_right_lon );
      geo_coord::geo_coordinate ur( wi_upper_right_lat, wi_upper_right_lon );
      w_img.bounds.add( ur);

      double wi_lower_right_lat;
      rs.fetch( col_index++, wi_lower_right_lat );
      double wi_lower_right_lon;
      rs.fetch( col_index++, wi_lower_right_lon );
      geo_coord::geo_coordinate lr( wi_lower_right_lat, wi_lower_right_lon );
      w_img.bounds.add( lr);

      double wi_lower_left_lat;
      rs.fetch( col_index++, wi_lower_left_lat );
      double wi_lower_left_lon;
      rs.fetch( col_index++, wi_lower_left_lon );
      geo_coord::geo_coordinate ll( wi_lower_left_lat, wi_lower_left_lon );
      w_img.bounds.add( ll );
      w_img_list.push_back( w_img );

    }
  }
  catch(cppdb::cppdb_error const &e)
  {
    LOG_ERROR( e.what() );
    CLOSE_DB_CONN;
    return false;
  }

  CLOSE_DB_CONN;
  return true;
}


bool
detection_db
::get_weighted_image_by_id( vxl_int_32 image_id, weighted_image& w_img )
{
  OPEN_DB_CONN;

  int col_index = 0;
  std::stringstream query;
  query << "select wi_image, wi_uuid, wi_frame_number, wi_frame_time, ";

  query << "st_y(wi_upper_left) as ul_lat, ";
  query << "st_x(wi_upper_left) as ul_lon, ";

  query << "st_y(wi_upper_right) as ur_lat, ";
  query << "st_x(wi_upper_right) as ur_lon, ";

  query << "st_y(wi_lower_right) as lr_lat, ";
  query << "st_x(wi_lower_right) as lr_lon, ";

  query << "st_y(wi_lower_left) as ll_lat, ";
  query << "st_x(wi_lower_left) as ll_lon ";

  query << "from weighted_image where wi_id = ?";

  try
  {
    cppdb::statement stmt = db_conn_.prepare( query.str() );
    stmt.bind( 1, image_id );

    std::vector<float> img;
    cppdb::result rs = stmt.query();

    // There is only 1 entry with a given ID
    if (rs.next())
    {
      std::stringstream img_stream;

      rs.fetch( col_index++, img_stream);
      if (img_stream)
      {
        img_stream.peek();
        if( img_stream.good() )
        {
          vsl_b_istream biss(&img_stream);
          if (img_stream.good())
          {
            vnl_matrix<float> M;
            vsl_b_read(biss, M);
            w_img.image = M;
          }
        }
      }

      std::string uuid_str;
      rs.fetch( col_index++, uuid_str );
      std::stringstream uuid_strstrm ( uuid_str );
      uuid_strstrm >> w_img.wi_uuid;

      unsigned frame_number;
      rs.fetch( col_index++, frame_number );

      double frame_time;
      rs.fetch( col_index++, frame_time );
      timestamp ts(frame_time, frame_number);
      w_img.time = ts;

      double wi_upper_left_lat;
      rs.fetch( col_index++, wi_upper_left_lat );
      double wi_upper_left_lon;
      rs.fetch( col_index++, wi_upper_left_lon );
      geo_coord::geo_coordinate ul(wi_upper_left_lat, wi_upper_left_lon);
      w_img.bounds.add( ul);

      double wi_upper_right_lat;
      rs.fetch( col_index++, wi_upper_right_lat );
      double wi_upper_right_lon;
      rs.fetch( col_index++, wi_upper_right_lon );
      geo_coord::geo_coordinate ur(wi_upper_right_lat, wi_upper_right_lon);
      w_img.bounds.add( ur);

      double wi_lower_right_lat;
      rs.fetch( col_index++, wi_lower_right_lat );
      double wi_lower_right_lon;
      rs.fetch( col_index++, wi_lower_right_lon );
      geo_coord::geo_coordinate lr(wi_lower_right_lat, wi_lower_right_lon);
      w_img.bounds.add( lr);

      double wi_lower_left_lat;
      rs.fetch( col_index++, wi_lower_left_lat );
      double wi_lower_left_lon;
      rs.fetch( col_index++, wi_lower_left_lon );
      geo_coord::geo_coordinate ll(wi_lower_left_lat, wi_lower_left_lon);
      w_img.bounds.add( ll);
    }
  }
  catch(cppdb::cppdb_error const &e)
  {
    LOG_ERROR( e.what() );
    CLOSE_DB_CONN;
    return false;
  }

  CLOSE_DB_CONN;
  return true;
}

bool
detection_db
::get_weighted_image_by_time( timestamp query_ts, std::vector<weighted_image>& w_img_list )
{
  OPEN_DB_CONN;

  std::stringstream query;
  query << "select wi_image, wi_uuid, wi_frame_number, wi_frame_time, ";

  query << "st_y(wi_upper_left) as ul_lat, ";
  query << "st_x(wi_upper_left) as ul_lon, ";

  query << "st_y(wi_upper_right) as ur_lat, ";
  query << "st_x(wi_upper_right) as ur_lon, ";

  query << "st_y(wi_lower_right) as lr_lat, ";
  query << "st_x(wi_lower_right) as lr_lon, ";

  query << "st_y(wi_lower_left) as ll_lat, ";
  query << "st_x(wi_lower_left) as ll_lon ";

  query << "from weighted_image where wi_session_id = ? and wi_frame_time = ?";

  try
  {
    cppdb::statement stmt = db_conn_.prepare( query.str() );
    stmt.bind( 1, this->session_id_ );
    stmt.bind( 2, query_ts.time() );

    std::vector<float> img;
    cppdb::result rs = stmt.query();

    // There is only 1 entry with a given ID
    while (rs.next())
    {
      weighted_image w_img;
      int col_index = 0;
      std::stringstream img_stream;

      rs.fetch( col_index++, img_stream);
      if (img_stream)
      {
        img_stream.peek();
        if( img_stream.good() )
        {
          vsl_b_istream biss(&img_stream);
          if (img_stream.good())
          {
            vnl_matrix<float> M;
            vsl_b_read(biss, M);
            w_img.image = M;
          }
        }
      }

      std::string uuid_str;
      rs.fetch( col_index++, uuid_str );
      std::stringstream uuid_strstrm ( uuid_str );
      uuid_strstrm >> w_img.wi_uuid;

      unsigned frame_number;
      rs.fetch( col_index++, frame_number );

      double frame_time;
      rs.fetch( col_index++, frame_time );
      timestamp ts( frame_time, frame_number );
      w_img.time = ts;

      double wi_upper_left_lat;
      rs.fetch( col_index++, wi_upper_left_lat );
      double wi_upper_left_lon;
      rs.fetch( col_index++, wi_upper_left_lon );
      geo_coord::geo_coordinate ul( wi_upper_left_lat, wi_upper_left_lon );
      w_img.bounds.add( ul);

      double wi_upper_right_lat;
      rs.fetch( col_index++, wi_upper_right_lat );
      double wi_upper_right_lon;
      rs.fetch( col_index++, wi_upper_right_lon );
      geo_coord::geo_coordinate ur( wi_upper_right_lat, wi_upper_right_lon );
      w_img.bounds.add( ur);

      double wi_lower_right_lat;
      rs.fetch( col_index++, wi_lower_right_lat );
      double wi_lower_right_lon;
      rs.fetch( col_index++, wi_lower_right_lon );
      geo_coord::geo_coordinate lr( wi_lower_right_lat, wi_lower_right_lon );
      w_img.bounds.add( lr);

      double wi_lower_left_lat;
      rs.fetch( col_index++, wi_lower_left_lat );
      double wi_lower_left_lon;
      rs.fetch( col_index++, wi_lower_left_lon );
      geo_coord::geo_coordinate ll( wi_lower_left_lat, wi_lower_left_lon );
      w_img.bounds.add( ll );
      w_img_list.push_back( w_img );

    }
  }
  catch(cppdb::cppdb_error const &e)
  {
    LOG_ERROR( e.what() );
    CLOSE_DB_CONN;
    return false;
  }

  CLOSE_DB_CONN;
  return true;
}


// ----------------------------------------------------------------------------
bool
detection_db
::add_detection( const image_object_sptr image_object, timestamp const& time )
{
  OPEN_DB_CONN;
  cppdb::transaction guard(db_conn_);
  try
  {
    add_detection_internal( image_object, time, "" );
    guard.commit();
    CLOSE_DB_CONN;
    return true;
  }
  catch(cppdb::cppdb_error const &e)
  {
    guard.rollback();
    LOG_ERROR( e.what() );
    CLOSE_DB_CONN;
    return false;
  }
}

// ----------------------------------------------------------------------------
bool
detection_db
::add_detections( std::vector<image_object_sptr> const& image_objects, timestamp const& time )
{
  return add_grouped_detections( image_objects, time, "");
}

bool
detection_db
::add_grouped_detections(
  std::vector<image_object_sptr> const& image_objects,
  timestamp const& time, std::string const& label)
{
  OPEN_DB_CONN;
  cppdb::transaction guard(db_conn_);
  try
  {
    std::vector<image_object_sptr>::const_iterator iter = image_objects.begin();
    for (; iter != image_objects.end(); ++iter)
    {
      add_detection_internal( *iter, time, label );
    }

    guard.commit();
    CLOSE_DB_CONN;
    return true;
  }
  catch(cppdb::cppdb_error const &e)
  {
    guard.rollback();
    LOG_ERROR( e.what() );
    CLOSE_DB_CONN;
    return false;
  }
}


// ----------------------------------------------------------------------------
void
detection_db
::add_detection_internal(
  const image_object_sptr img_obj,
  timestamp const& time, std::string const& label )
{
  // not storing boundary at the moment
  std::stringstream insert_img_obj_sql;
  insert_img_obj_sql << "insert into " << DETECTION_TABLE_NAME;
  insert_img_obj_sql << INSERT_DETECTIONS_COLUMN_LIST;
  insert_img_obj_sql << "values (?,?,?,?,?, ";
  insert_img_obj_sql << SPATIAL_GENERIC_POINT_INSERT_STMT << ",";
  insert_img_obj_sql << SPATIAL_GENERIC_POINT_INSERT_STMT << ",";
  insert_img_obj_sql << SPATIAL_LON_LAT_POINT_INSERT_STMT << ",";
  insert_img_obj_sql << SPATIAL_LON_LAT_POLYGON_INSERT_STMT << ",";
  insert_img_obj_sql << "  ?,?,?,?,?,?,?,?,?, ";
  insert_img_obj_sql << SPATIAL_GENERIC_POINT_INSERT_STMT << ",";
  insert_img_obj_sql << SPATIAL_GENERIC_POINT_INSERT_STMT << ",";
  insert_img_obj_sql << "  ?,?,?,?);";

  cppdb::statement img_obj_stmt = db_conn_.prepare( insert_img_obj_sql.str() );

  int col_index = 1;

  //bind the track_session_id for the state.
  //Makes certain queries more efficient.
  img_obj_stmt.bind( col_index++, this->session_id_ );

  //bind the frame_number
  unsigned frame_number = 0;
  double frame_time = 0;
  if (time.is_valid())
  {
    frame_number = time.frame_number();
    frame_time = time.time();
  }

  img_obj_stmt.bind( col_index++, frame_number );

  //bind frame_time
  img_obj_stmt.bind( col_index++, frame_time );

  img_obj_stmt.bind( col_index++, img_obj->get_area() );
  img_obj_stmt.bind( col_index++, img_obj->get_image_area() );

  tracker_world_coord_type const& world_loc = img_obj->get_world_loc();

  //bind world loc
  img_obj_stmt.bind( col_index++, world_loc[0] );
  img_obj_stmt.bind( col_index++, world_loc[1] );

  vidtk_pixel_coord_type const& img_loc = img_obj->get_image_loc();

  //bind img_loc
  img_obj_stmt.bind( col_index++, img_loc[0] );
  img_obj_stmt.bind( col_index++, img_loc[1] );

  //bind lonlat
  geo_coord::geo_coordinate geo_ = img_obj->get_geo_loc();
  img_obj_stmt.bind( col_index++, geo_.get_longitude() );
  img_obj_stmt.bind( col_index++, geo_.get_latitude() );

  vgl_box_2d<unsigned> const& img_bbox = img_obj->get_bbox();

  //connect 5 points of the bbox to make a complete region.
  img_obj_stmt.bind( col_index++, img_bbox.min_x());
  img_obj_stmt.bind( col_index++, img_bbox.min_y());

  img_obj_stmt.bind( col_index++, img_bbox.max_x());
  img_obj_stmt.bind( col_index++, img_bbox.min_y());

  img_obj_stmt.bind( col_index++, img_bbox.max_x());
  img_obj_stmt.bind( col_index++, img_bbox.max_y());

  img_obj_stmt.bind( col_index++, img_bbox.min_x());
  img_obj_stmt.bind( col_index++, img_bbox.max_y());

  img_obj_stmt.bind( col_index++, img_bbox.min_x());
  img_obj_stmt.bind( col_index++, img_bbox.min_y());

  // Extract the image as a binary blob
  vil_image_resource_sptr view;
  unsigned int border;

  if( img_obj->get_image_chip(view, border) )
  {
    std::stringstream img_stream;

    vsl_b_ostream bss( &img_stream );
    write_img_resource_b( bss, view );
    img_stream.flush();

    std::stringstream bind_stream( img_stream.str() );
    img_obj_stmt.bind( col_index++, bind_stream );

    //bind image_chip offset
    if ( border != INVALID_IMG_CHIP_BORDER )
    {
      img_obj_stmt.bind( col_index++, border );
    }
    else
    {
      ++col_index;
    }
  }
  else
  {
    /// increment once for the image chip and once for the border
    col_index += 2;
  }

  vil_image_view<bool> mask;
  image_object::image_point_type mask_origin;

  if ( img_obj->get_object_mask( mask, mask_origin ) )
  {
    std::stringstream mask_stream;

    vsl_b_ostream mask_bss(&mask_stream);
    vsl_b_write(mask_bss, mask);
    mask_stream.flush();

    std::stringstream bind_stream(mask_stream.str());
    img_obj_stmt.bind( col_index++, bind_stream);

    //bind mask
    img_obj_stmt.bind( col_index++, mask_origin.x() );
    img_obj_stmt.bind( col_index++, mask_origin.y() );
  }
  else
  {
    col_index += 3;
  }

  image_histogram hist;
  if (img_obj->get_histogram( hist ))
  {
    std::stringstream hist_stream;
    vil_image_view<double> hist_view = hist.get_h();

    vsl_b_ostream hist_bss( &hist_stream );
    vsl_b_write( hist_bss, hist_view );
    hist_stream.flush();
    std::stringstream bind_stream( hist_stream.str() );

    //bind hist
    img_obj_stmt.bind( col_index++, bind_stream);
    img_obj_stmt.bind( col_index++, hist.type() );
    img_obj_stmt.bind( col_index++, hist.mass() );

    //    std::string hist_cube = histogram_to_cube_string( hist_view );
    //    img_obj_stmt.bind( col_index++, hist_cube );
  }
  else
  {
    col_index += 3;
  }

  image_object::heat_map_sptr heatmap;
  vgl_point_2d<unsigned> map_origin;
  if (img_obj->get_heat_map( heatmap, map_origin ))
  {
    std::stringstream hm_stream;

    vsl_b_ostream hm_bss(&hm_stream);
    vsl_b_write(hm_bss, heatmap.get());
    hm_stream.flush();
    std::stringstream bind_stream(hm_stream.str());

    //bind heat map
    img_obj_stmt.bind( col_index++, bind_stream);
    img_obj_stmt.bind( col_index++, map_origin.x() );
    img_obj_stmt.bind( col_index++, map_origin.y() );
  }
  else
  {
    col_index += 3;
  }

  image_object::intensity_distribution_type dist;
  img_obj->get_intensity_distribution( dist );
  img_obj_stmt.bind( col_index++, dist.first );
  img_obj_stmt.bind( col_index++, dist.second );

  double confidence = img_obj->get_confidence();
  img_obj_stmt.bind( col_index++, confidence);

  image_object::source_code type = img_obj->get_source_type();
  std::string name = img_obj->get_source_name();
  img_obj_stmt.bind( col_index++, type );
  img_obj_stmt.bind( col_index++, name );

  if (!label.empty() )
  {
    img_obj_stmt.bind (col_index++, label );
  }
  else
  {
    col_index++;
  }

  img_obj_stmt.exec();
}


// ----------------------------------------------------------------------------
std::vector <image_object_sptr> const
detection_db
::get_all_detections()
{
  std::vector <image_object_sptr> objects;

  try
  {
    OPEN_DB_CONN;
    std::stringstream select_img_obj_sql;
    select_img_obj_sql
      << SELECT_DET_COLUMNS << " "
      << "from image_object where " << DET_SESSION_ID_COL << "= ? "
      << "order by det_frame_time";

    cppdb::statement img_obj_stmt = db_conn_.create_statement ( select_img_obj_sql.str() );
    img_obj_stmt.bind( 1, this->session_id_);

    cppdb::result rs = img_obj_stmt.query();

    while (rs.next ())
    {
      objects.push_back (load_image_object (rs));
    }
  }
  catch (cppdb::cppdb_error const &e)
  {
    LOG_ERROR( e.what() );
  }

  CLOSE_DB_CONN;
  return objects;
}


// ----------------------------------------------------------------------------
std::vector <std::pair <image_object_sptr, double> > const
detection_db
::get_all_detections_by_distance( std::string const& dist_func, std::string const& hist_data, int N )
{

  /*
    SELECT *,
    histogram_cube
    <#>
    '(0, 0, 0, 0, 0, 0.0416667, 0.888889, 0.0694444, 0, 0, 0, 0, 0, 0, 0)'::cube as dist
    FROM image_object
    ORDER BY
    histogram_cube
    <#>
    '(0, 0, 0, 0, 0, 0.0416667, 0.888889, 0.0694444, 0, 0, 0, 0, 0, 0, 0)'::cube limit 10;

  */

  /*
    <#> = distance_taxicab
    <-> = cube_distance (Euclidian?)
    <=> = distance_chebyshev

   */
  std::vector <std::pair <image_object_sptr, double> > objects;

  try
  {
    OPEN_DB_CONN;
    std::stringstream select_img_obj_sql;
    select_img_obj_sql
      << SELECT_DET_COLUMNS << "," << DET_HISTOGRAM_CUBE_COL << " "
      << dist_func << " " << hist_data << "::cube as dist "
      << "from image_object where " << DET_SESSION_ID_COL << " = ? "
      << "order by " DET_HISTOGRAM_CUBE_COL << dist_func << " "
      << hist_data << "::cube limit " << N;

    LOG_DEBUG(select_img_obj_sql.str());

    cppdb::statement img_obj_stmt = db_conn_.create_statement ( select_img_obj_sql.str() );
    img_obj_stmt.bind( 1, this->session_id_);

    cppdb::result rs = img_obj_stmt.query();

    while (rs.next ())
    {
      double dist = 0.0;
      rs.fetch( "dist", dist);
      objects.push_back ( std::make_pair (load_image_object (rs), dist));
    }
  }
  catch (cppdb::cppdb_error const &e)
  {
    LOG_ERROR( e.what() );
  }

  CLOSE_DB_CONN;
  return objects;
}



// ----------------------------------------------------------------------------
std::vector <image_object_sptr> const
detection_db
::get_all_detections_by_temporal_intersect(
  const timestamp& st_time, const timestamp& end_time )
{
  std::vector <image_object_sptr> objects;

  try
  {
    OPEN_DB_CONN;
    std::stringstream select_img_obj_sql;
    select_img_obj_sql
      << SELECT_DET_COLUMNS
      << "from " << DETECTION_TABLE_NAME << " "
      << "where " << DET_SESSION_ID_COL << " = ? "
      << "and " << DET_FRAME_TIME_COL << " >= " << st_time.time() << " "
      << "and " << DET_FRAME_TIME_COL << " <= " << end_time.time() << " "
      << "order by " << DET_FRAME_TIME_COL;

    //  LOG_DEBUG(select_img_obj_sql.str());

    cppdb::statement img_obj_stmt = db_conn_.create_statement ( select_img_obj_sql.str() );
    img_obj_stmt.bind( 1, this->session_id_);

    cppdb::result rs = img_obj_stmt.query();

    while (rs.next ())
    {
      objects.push_back (load_image_object (rs));
    }
  }
  catch (cppdb::cppdb_error const &e)
  {
    LOG_ERROR( e.what() );
  }

  CLOSE_DB_CONN;
  return objects;
}

// ----------------------------------------------------------------------------
std::vector <image_object_sptr> const
detection_db
::get_all_detections_geo_intersect( vgl_polygon <float_type> const& region )
{
  std::vector <image_object_sptr> objects;

  try
  {
    OPEN_DB_CONN;
    assert( region.num_sheets() == 1 );
    vgl_polygon<float_type>::sheet_t const& sheet = region[0];

    std::size_t const n = sheet.size();
    assert( n > 0 );


    std::stringstream point_stream;
    point_stream.precision(16);
    double firstX, firstY = 0;
    for( unsigned i = 0; i < n; ++i )
    {
      if (i == 0 )
      {
        firstX = sheet[i].x();
        firstY = sheet[i].y();
      }
      point_stream << sheet[i].x() << "  ";
      point_stream << sheet[i].y() << "  ";
      point_stream << " , ";
    }
    point_stream << firstX << " " << firstY;

    std::stringstream select_img_obj_sql;
    select_img_obj_sql
      << SELECT_DET_COLUMNS
      << "from " << DETECTION_TABLE_NAME << " "
      << "where " << DET_SESSION_ID_COL << " = ? and "
      << "( ST_Within(" << DET_LONLAT_COL << ","
      << "ST_GeomFromText('POLYGON((" << point_stream.str() << " ))', 4326 ))) "
      << "order by " << DET_FRAME_TIME_COL;

    cppdb::statement img_obj_stmt = db_conn_.create_statement ( select_img_obj_sql.str() );
    img_obj_stmt.bind( 1, this->session_id_);

    cppdb::result rs = img_obj_stmt.query();

    while (rs.next ())
    {
      objects.push_back (load_image_object (rs));

    }
  }
  catch (cppdb::cppdb_error const &e)
  {
    LOG_ERROR( e.what() );
  }

  CLOSE_DB_CONN;
  return objects;
}

// ----------------------------------------------------------------------------
image_object_sptr
detection_db
::load_image_object( cppdb::result & rs )
{
  image_object_sptr img_obj = new image_object();
  int col_index = 0;

  int session_id;
  rs.fetch( col_index++, session_id);

  //track_state timestamp()
  unsigned frame_num;
  rs.fetch( col_index++, frame_num );

  double frame_time;
  rs.fetch( col_index++, frame_time );

  /// @todo: what to do with the timestamp, put in image_object?
  //  timestamp ts = timestamp(frame_time, frame_num);

  double area;
  rs.fetch( col_index++, area );
  img_obj->set_area( area );

  double img_area;
  rs.fetch( col_index++, img_area );
  img_obj->set_image_area( img_area );

  //track_state world loc x & y
  double world_loc_x, world_loc_y;
  rs.fetch( col_index++, world_loc_x );
  rs.fetch( col_index++, world_loc_y );
  img_obj->set_world_loc( world_loc_x, world_loc_y, 0 );

  //track_state img loc
  double img_loc_x, img_loc_y;
  rs.fetch( col_index++, img_loc_x );
  rs.fetch( col_index++, img_loc_y );
  img_obj->set_image_loc( img_loc_x, img_loc_y );

  double lon, lat;
  rs.fetch( col_index++, lon );
  rs.fetch( col_index++, lat );
  geo_coord::geo_coordinate geo(lat, lon);
  img_obj->set_geo_loc( geo );

  unsigned min_x, min_y, max_x, max_y;
  rs.fetch( col_index++, min_x );
  rs.fetch( col_index++, min_y );
  rs.fetch( col_index++, max_x );
  rs.fetch( col_index++, max_y );

  vgl_box_2d<unsigned> bbox(
    vgl_point_2d<unsigned int>(min_x, min_y),
    vgl_point_2d<unsigned int>(max_x, max_y) );
  img_obj->set_bbox( bbox );

  //get image_chip
  {
    vil_image_resource_sptr img;
    unsigned image_chip_offset = INVALID_IMG_CHIP_BORDER;

    std::stringstream blob_stream;
    bool offset_read = false;
    rs.fetch( col_index++, blob_stream );

    if (blob_stream)
    {
      blob_stream.peek();

      if( blob_stream.good() )
      {
        vsl_b_istream biss(&blob_stream);
        if( blob_stream.good() )
        {
          read_img_resource_b(biss, img);
        }
        if( blob_stream.good() )
        {
          bool has_value = rs.fetch( col_index++, image_chip_offset );
          offset_read = true;
          if ( !has_value )
          {
            // we didn't get anything back from db, so initialize to invalid.
            image_chip_offset = INVALID_IMG_CHIP_BORDER;
          }

          img_obj->set_image_chip(img, image_chip_offset);
        }
      }
    }
    // skip over the offset since we didn't read it.
    if (!offset_read)
    {
      ++col_index;
    }
  }

  //get image_mask
  {
    vil_image_view<bool> img;
    std::stringstream blob_stream;

    // this column immediately proceeds the image_chip column in all queries
    bool mask_read = false;
    rs.fetch( col_index++, blob_stream );
    if (blob_stream)
    {
      blob_stream.peek();

      if( blob_stream.good() )
      {
        vsl_b_istream biss(&blob_stream);
        if( blob_stream.good() )
        {
          vsl_b_read(biss, img);
        }
        if( blob_stream.good() )
        {
          unsigned mask_i0 = 0;
          unsigned mask_j0 = 0;
          bool has_value = rs.fetch( col_index++, mask_i0 );
          if ( !has_value )
          {
            mask_i0 = 0;
          }

          has_value = rs.fetch( col_index++, mask_j0 );
          if ( !has_value )
          {
            mask_j0 = 0;
          }
          img_obj->set_object_mask( img, image_object::image_point_type(mask_i0, mask_j0) );
          mask_read = true;
        }
      }
    }
    // didn't read the mask, so skip both columns
    if (!mask_read)
    {
      col_index += 2;
    }
  }

  //get image histogram
  {
    image_histogram hist;
    Image_Histogram_Type hist_type;
    double hist_mass;
    vil_image_view<double> img;

    std::stringstream blob_stream;
    bool hist_read = false;
    rs.fetch( col_index++, blob_stream );

    if (blob_stream)
    {
      blob_stream.peek();

      if( blob_stream.good() )
      {
        vsl_b_istream biss(&blob_stream);
        if( blob_stream.good() )
        {
          vsl_b_read( biss, img );
        }
        if( blob_stream.good() )
        {
          // histogram is good, so get type and mass
          int ht;
          rs.fetch( col_index++, ht );
          hist_type = static_cast<vidtk::Image_Histogram_Type>(ht);

          rs.fetch( col_index++, hist_mass );
          hist.set_type( hist_type );
          hist.set_mass( hist_mass );
          hist.set_h( img );
          img_obj->set_histogram( hist );
          hist_read = true;
        }
      }
    }
    // didn't read histogram details, skip both columns
    if (!hist_read)
    {
      col_index += 2;
    }
  }

  //get heat map
  {
    vil_image_view<float> map;
    std::stringstream blob_stream;

    bool heatmap_read = false;
    rs.fetch( col_index++, blob_stream );

    if (blob_stream)
    {
      blob_stream.peek();

      if( blob_stream.good() )
      {
        vsl_b_istream biss(&blob_stream);
        if( blob_stream.good() )
        {
          vsl_b_read(biss, map);
        }
        if( blob_stream.good() )
        {
          double origin_x, origin_y;

          rs.fetch( col_index++, origin_x );
          rs.fetch( col_index++, origin_y );

          vgl_point_2d<unsigned> origin (origin_x, origin_y);
          heatmap_read = true;
          ///@ todo: figure out how to store the heatmap in image_object
          //::vidtk::image_object::heat_map_sptr map = boost::make_shared< ::vidtk::image_object::heat_map_type >();
          //          image_object::heat_map_sptr heatmap = map;
          //          img_obj->set_heat_map( heatmap, origin );
        }
      }
    }
    if (!heatmap_read)
    {
      col_index += 2;
    }
  }

  double intense_dist_x, intense_dist_y;

  rs.fetch( col_index++, intense_dist_x );
  rs.fetch( col_index++, intense_dist_y );
  std::pair<float, float> dist(intense_dist_x, intense_dist_y);
  img_obj->set_intensity_distribution( dist );

  double confidence;
  rs.fetch( col_index++, confidence );
  img_obj->set_confidence( confidence );

  std::string source_instance_name;
  int source_instance_type;
  rs.fetch( col_index++, source_instance_type );
  rs.fetch( col_index++, source_instance_name );
  img_obj->set_source( static_cast<image_object::source_code>( source_instance_type ),  source_instance_name );

  return img_obj;
}

std::string
detection_db
::histogram_to_cube_string (vil_image_view<double> const& hist_view)
{
  std::stringstream hist_cube_stream;

  // Add the histogram as a cube.
  hist_cube_stream << "cube(array[";
  double* dbl_chnk = static_cast<double*>(hist_view.memory_chunk()->data());
  for (size_t i = 0; i < hist_view.size() - 2; ++i)
  {
    double d = dbl_chnk[i];
    hist_cube_stream << d << ",";
    //    double d = static_cast<double>(hist_view.memory_chunk()->data()[i]);
    //      hist_stream << hist_view.memory_chunk()->data()[i];//hg_chunk[i];
  }

  hist_cube_stream << dbl_chnk[hist_view.size() - 1];
  hist_cube_stream << "])";

  return hist_cube_stream.str();
}


void
detection_db
::set_active_session_id(vxl_int_32 s_id)
{
  this->session_id_ = s_id;
}

} //namespace vidtk
