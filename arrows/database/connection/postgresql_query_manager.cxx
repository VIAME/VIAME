/*ckwg +5
 * Copyright 2012-2015 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "postgresql_query_manager.h"

namespace vidtk {

const std::string postgresql_query_manager::POSTGIS_LINESTRING_INSERT_STMT =
  "ST_GeomFromText('LINESTRING(' || ? || ')')";
const std::string postgresql_query_manager::POSTGIS_POINT_INSERT_STMT =
  "ST_GeomFromText('POINT(' || ? || ' ' || ? || ')')";
const std::string postgresql_query_manager::POSTGIS_POLYGON_INSERT_STMT =
  "ST_GeomFromText('POLYGON(('\
   || ? || ' ' || ? || ',' || ? || ' ' || ? || ','\
   || ? || ' ' || ? || ',' || ? || ' ' || ? || ','\
   || ? || ' ' || ? || '))')";


postgresql_query_manager
::postgresql_query_manager()
{

}

postgresql_query_manager
::~postgresql_query_manager()
{

}

const std::string &
postgresql_query_manager
::insert_track_query() const
{
  static std::string POSTGRESQL_INSERT_TRACK_SQL =
    "INSERT INTO " + TRACK_TABLE_NAME +
    " ( " + TRACK_UUID_COL            +
    " , " + TRACK_EXT_ID_COL          +
    " , " + TRACK_SESSION_ID_COL      +
    " , " + TRACK_LAST_MOD_MATCH_COL  +
    " , " + TRACK_FALSE_ALARM_COL     +
    " , " + TRACKER_ID_COL            +
    " , " + TRACKER_TYPE_COL          +
    " , " + TRACK_ATTRS_COL           +
    " , " + TRACK_REGION_COL          +
    " ) VALUES(?,?,?,?,?,?,?,?," + POSTGIS_LINESTRING_INSERT_STMT + ")";

  return POSTGRESQL_INSERT_TRACK_SQL;
}


const std::string &
postgresql_query_manager
::insert_track_state_query() const
{
  static std::string POSTGRESQL_INSERT_TRACK_STATE_SQL =
    "INSERT INTO " + STATE_TABLE_NAME   +
    " ( " + STATE_TRACK_ID_COL          +
    " , " + STATE_SESSION_ID_COL        +
    " , " + STATE_FRAME_NUM_COL         +
    " , " + STATE_FRAME_TIME_COL        +
    " , " + STATE_LOC_COL               +
    " , " + STATE_WORLD_LOC_COL         +
    " , " + STATE_IMG_LOC_COL           +
    " , " + STATE_LONLAT_COL            +

    " , " + STATE_UTM_SMOOTHED_LOC_E_COL   +
    " , " + STATE_UTM_SMOOTHED_LOC_N_COL   +
    " , " + STATE_UTM_SMOOTHED_LOC_ZONE_COL +
    " , " + STATE_UTM_SMOOTHED_LOC_IS_NORTH_COL +

    " , " + STATE_UTM_RAW_LOC_E_COL       +
    " , " + STATE_UTM_RAW_LOC_N_COL       +
    " , " + STATE_UTM_RAW_LOC_ZONE_COL    +
    " , " + STATE_UTM_RAW_LOC_ZONE_IS_NORTH_COL +

    " , " + STATE_UTM_VELOCITY_X_COL       +
    " , " + STATE_UTM_VELOCITY_Y_COL       +

    " , " + STATE_VELOCITY_X_COL        +
    " , " + STATE_VELOCITY_Y_COL        +
    " , " + STATE_BBOX_COL              +
    " , " + STATE_ATTRS_COL             +
    " , " + STATE_AREA_COL              +
    " , " + STATE_IMAGE_CHIP_COL        +
    " , " + STATE_IMAGE_CHIP_OFFSET_COL +
    " , " + STATE_IMAGE_MASK_COL        +
    " , " + STATE_MASK_i0_COL           +
    " , " + STATE_MASK_j0_COL           +
    " , " + STATE_COV_R0C0_COL          +
    " , " + STATE_COV_R0C1_COL          +
    " , " + STATE_COV_R1C0_COL          +
    " , " + STATE_COV_R1C1_COL          +

    " ) VALUES (?,?,?,?"
    "," + POSTGIS_POINT_INSERT_STMT +
    "," + POSTGIS_POINT_INSERT_STMT +
    "," + POSTGIS_POINT_INSERT_STMT +
    "," + POSTGIS_POINT_INSERT_STMT +
    ",?,?,?,?,?,?,?,?,?,?,?,?"
    "," + POSTGIS_POLYGON_INSERT_STMT +
    ",?,?,?,?,?,?,?,?,?,?,?)";

  return POSTGRESQL_INSERT_TRACK_STATE_SQL;
}

} // namespace vidtk
