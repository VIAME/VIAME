/*ckwg +5
 * Copyright 2016-2017 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef vidtk_detection_db_
#define vidtk_detection_db_

#include <cppdb/frontend.h>
#include <database/vidtk_database.h>
#include <utilities/geo_bounds.h>
#include <utilities/geo_coordinate.h>
#include <utilities/timestamp.h>

#include <tracking_data/image_object.h>

#include <vnl/vnl_matrix.h>
//#include <cppdb/pool.h>
#include <database/detection_db_defs.h>

namespace vidtk
{


struct weighted_image
{
  uuid_t wi_uuid;
  vnl_matrix<float> image;
  timestamp time;
  geo_coord::geo_bounds bounds;
};

class detection_db
{

public
:
  detection_db( db_connection_params const& params );
  virtual ~detection_db();

  vxl_int_32 add_weighted_image( weighted_image const& );
  bool get_weighted_image_by_id( vxl_int_32 image_id, weighted_image& );
  bool get_all_weighted_images( std::vector<weighted_image>& w_img_list );
  bool get_weighted_image_by_time( timestamp ts, std::vector<weighted_image>& w_img_list );

  /**
   * \brief Add a detection to the database.
   *
   * @param The image_object to add.
   * @param The timestamp for this object.
   *
   * @return \b true if image_object is added; \b false otherwise.
   */
  bool add_detection( const image_object_sptr, const timestamp& );
  bool add_detections( std::vector<image_object_sptr> const&, const timestamp& );

  bool add_grouped_detections( std::vector<image_object_sptr> const&, const timestamp& , std::string const& label);

    /// Get all detections for a session
  const std::vector <image_object_sptr> get_all_detections();

  std::vector <std::pair <image_object_sptr, double> > const
  get_all_detections_by_distance( std::string const& dist_func, std::string const& hist_data, int N );

  /// Get all detections for a time range
  const std::vector <image_object_sptr>
  get_all_detections_by_temporal_intersect(
    timestamp const& st_time, timestamp const& end_time );

  /// Get all detections that intersect a polygon
  const std::vector <image_object_sptr>
  get_all_detections_geo_intersect( vgl_polygon <float_type> const& region );

  void set_active_session_id(vxl_int_32 s_id);

private:
  void add_detection_internal(
    const image_object_sptr img_obj, timestamp const& time, std::string const& label );
  image_object_sptr load_image_object( cppdb::result & rs );
  std::string histogram_to_cube_string (vil_image_view<double> const& hist_view);

  cppdb::session db_conn_;
  std::string connect_string_;
  vxl_int_32 session_id_;

  //  cppdb::pool::pointer db_conn_pool_;

}; // class detection_db


} // namespace vidtk

#endif // vidtk_detection_db_
