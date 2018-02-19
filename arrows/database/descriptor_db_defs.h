

namespace kwiver {
namespace arrows {
namespace database {

#define OPEN_DB_CONN db_conn_.open( connect_string_ );
#define CLOSE_DB_CONN db_conn_.close();

/*
#define DETECTION_TABLE_NAME "image_object "
#define DET_ID "img_obj_id"
#define DET_SESSION_ID_COL "det_session_id "
#define DET_FRAME_NUM_COL "det_frame_num "
#define DET_FRAME_TIME_COL "det_frame_time "
#define DET_AREA_COL "det_area "
#define DET_IMAGE_AREA_COL "det_image_area "
#define DET_WORLD_LOC_COL "det_world_loc "
#define DET_IMG_LOC_COL "det_img_loc "
#define DET_LONLAT_COL "det_lonlat "
#define DET_BBOX_COL "det_bbox "
#define DET_IMAGE_CHIP_COL "det_image_chip "
#define DET_IMAGE_CHIP_OFFSET_COL "det_image_chip_offset "
#define DET_IMAGE_MASK_COL "det_image_mask "
#define DET_MASK_I0_COL "det_mask_i0 "
#define DET_MASK_J0_COL "det_mask_j0 "
#define DET_IMAGE_HISTOGRAM_COL "det_image_histogram "
#define DET_HISTOGRAM_TYPE_COL "det_histogram_type "
#define DET_HISTOGRAM_MASS_COL "det_histogram_mass "
#define DET_HEAT_MAP_COL "det_heat_map "
#define DET_HEAT_MAP_ORIGIN_COL "det_heat_map_origin "
#define DET_INTENSITY_DIST_COL "det_intensity_dist "
#define DET_CONFIDENCE_COL "det_confidence "
#define DET_SOURCE_INSTANCE_TYPE_COL "det_source_instance_type "
#define DET_SOURCE_INSTANCE_NAME_COL "det_source_instance_name "
#define DET_HISTOGRAM_CUBE_COL "det_histogram_cube"
#define DET_WORLD_LOC_X "st_x(" DET_WORLD_LOC_COL ") as wl_x "
#define DET_WORLD_LOC_Y "st_Y(" DET_WORLD_LOC_COL ") as wl_Y "
#define DET_IMG_LOC_X "st_x(" DET_IMG_LOC_COL ") as il_x "
#define DET_IMG_LOC_Y "st_Y(" DET_IMG_LOC_COL ") as il_y "
#define DET_LONLAT_X "st_x(" DET_LONLAT_COL ") as lon "
#define DET_LONLAT_Y "st_Y(" DET_LONLAT_COL ") as lat "

#define DET_BBOX_UL_X "st_x(st_pointN(ST_Boundary( " DET_BBOX_COL "),1)) "
#define DET_BBOX_UL_Y "st_y(st_pointN(ST_Boundary( " DET_BBOX_COL "),1)) "
#define DET_BBOX_LR_X "st_x(st_pointN(ST_Boundary( " DET_BBOX_COL "),3)) "
#define DET_BBOX_LR_Y "st_y(st_pointN(ST_Boundary( " DET_BBOX_COL "),3)) "
#define DET_BBOX_CORNERS DET_BBOX_UL_X "," DET_BBOX_UL_Y "," DET_BBOX_LR_X "," DET_BBOX_LR_Y " "

#define DET_HEATMAP_ORIGIN_X "st_x( " DET_HEAT_MAP_ORIGIN_COL ") as hmo_x "
#define DET_HEATMAP_ORIGIN_Y "st_y( " DET_HEAT_MAP_ORIGIN_COL ") as hmo_y "
#define DET_INTENSITY_DIST_X "st_x( " DET_INTENSITY_DIST_COL ") as id_x "
#define DET_INTENSITY_DIST_Y "st_y( " DET_INTENSITY_DIST_COL ") as id_y "

#define DET_GROUP_LABEL "det_group_label"

#define SELECT_DET_COLUMNS                                              \
  "select " DET_SESSION_ID_COL "," DET_FRAME_NUM_COL "," DET_FRAME_TIME_COL "," \
  DET_AREA_COL "," DET_IMAGE_AREA_COL "," DET_WORLD_LOC_X "," DET_WORLD_LOC_Y "," \
  DET_IMG_LOC_X "," DET_IMG_LOC_Y "," DET_LONLAT_X "," DET_LONLAT_Y "," \
  DET_BBOX_CORNERS "," DET_IMAGE_CHIP_COL "," DET_IMAGE_CHIP_OFFSET_COL "," \
  DET_IMAGE_MASK_COL "," DET_MASK_I0_COL "," DET_MASK_J0_COL "," \
  DET_IMAGE_HISTOGRAM_COL "," DET_HISTOGRAM_TYPE_COL "," DET_HISTOGRAM_MASS_COL "," \
  DET_HEAT_MAP_COL "," DET_HEATMAP_ORIGIN_X "," DET_HEATMAP_ORIGIN_Y "," \
  DET_INTENSITY_DIST_X "," DET_INTENSITY_DIST_Y "," DET_CONFIDENCE_COL "," \
  DET_SOURCE_INSTANCE_TYPE_COL "," DET_SOURCE_INSTANCE_NAME_COL

#define INSERT_DETECTIONS_COLUMN_LIST \
  "( " DET_SESSION_ID_COL "," DET_FRAME_NUM_COL "," DET_FRAME_TIME_COL "," \
  DET_AREA_COL "," DET_IMAGE_AREA_COL ","                               \
  DET_WORLD_LOC_COL "," DET_IMG_LOC_COL "," DET_LONLAT_COL ","          \
  DET_BBOX_COL "," DET_IMAGE_CHIP_COL "," DET_IMAGE_CHIP_OFFSET_COL "," \
  DET_IMAGE_MASK_COL "," DET_MASK_I0_COL "," DET_MASK_J0_COL ","        \
  DET_HISTOGRAM_TYPE_COL "," DET_IMAGE_HISTOGRAM_COL "," DET_HISTOGRAM_MASS_COL "," \
  DET_HEAT_MAP_COL "," DET_HEAT_MAP_ORIGIN_COL "," \
  DET_INTENSITY_DIST_COL "," DET_CONFIDENCE_COL "," DET_SOURCE_INSTANCE_TYPE_COL "," \
  DET_SOURCE_INSTANCE_NAME_COL "," DET_GROUP_LABEL ")"

#define DETECTION_INSERT_STMT \
  "insert into " DETECTION_TABLE_NAME                                   \
  INSERT_DETECTIONS_COLUMN_LIST                                         \
  "values (?,?,?,?,?, "                                                 \
  SPATIAL_GENERIC_POINT_INSERT_STMT  ","                                \
  SPATIAL_GENERIC_POINT_INSERT_STMT  ","                                \
  SPATIAL_LON_LAT_POINT_INSERT_STMT  ","                                \
  SPATIAL_LON_LAT_POLYGON_INSERT_STMT ","                               \
  "  ?,?,?,?,?,?,?,?,?, "                                             \
  SPATIAL_GENERIC_POINT_INSERT_STMT  ","                                \
  SPATIAL_GENERIC_POINT_INSERT_STMT  ","                                \
  "  ?,?,?,?);"

const std::string SPATIAL_LON_LAT_POINT_INSERT_STMT =
  "ST_GeomFromText('POINT(' || ? || ' ' || ? || ')', 4326)";

const std::string SPATIAL_GENERIC_POINT_INSERT_STMT =
  "ST_GeomFromText('POINT(' || ? || ' ' || ? || ')')";

const std::string SPATIAL_LON_LAT_POLYGON_INSERT_STMT =
  "ST_GeomFromText('POLYGON((' \
    || ? || ' ' || ? || ',' || ? || ' ' || ? || ',' \
    || ? || ' ' || ? || ',' || ? || ' ' || ? || ',' \
    || ? || ' ' || ? || '))') ";

*/
} } } // end namespace
