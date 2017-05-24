/*ckwg +5
 * Copyright 2012-2015 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "query_manager.h"
#include <vbl/vbl_smart_ptr.txx>
#include <boost/lexical_cast.hpp>

namespace vidtk {

const std::string query_manager::POLYGON = "'POLYGON(('\
   || ? || ' ' || ? || ',' || ? || ' ' || ? || ','\
   || ? || ' ' || ? || ',' || ? || ' ' || ? || ','\
   || ? || ' ' || ? || '))'";


//track_table definitions
const std::string query_manager::VIDTK_DB_SCHEMA_NAME = "";
const std::string query_manager::TRACK_TABLE_NAME = "TRACK";
const std::string query_manager::TRACK_ID_PK_COL = "TRACK_ID";
const std::string query_manager::TRACK_UUID_COL = "TRACK_UUID";
const std::string query_manager::TRACK_EXT_ID_COL = "TRACK_EXT_ID";
const std::string query_manager::TRACK_SESSION_ID_COL = "TRACK_SESSION_ID";
const std::string query_manager::TRACK_LAST_MOD_MATCH_COL = "LAST_MOD_MATCH";
const std::string query_manager::TRACK_FALSE_ALARM_COL = "FALSE_ALARM";
const std::string query_manager::TRACK_START_TIME_COL = "TRACK_START_TIME";
const std::string query_manager::TRACK_END_TIME_COL = "TRACK_END_TIME";
const std::string query_manager::TRACK_REGION_COL = "TRACK_REGION";
const std::string query_manager::TRACK_LOC_UPPER_LEFT_X_COL = "TRACK_LOC_UPPER_LEFT";
const std::string query_manager::TRACK_LOC_UPPER_LEFT_Y_COL = "TRACK_LOC_UPPER_LEFT_Y";
const std::string query_manager::TRACK_LOC_LOWER_RIGHT_X_COL = "TRACK_LOC_LOWER_RIGHT_X";
const std::string query_manager::TRACK_LOC_LOWER_RIGHT_Y_COL = "TRACK_LOC_LOWER_RIGHT_Y";
const std::string query_manager::TRACKER_ID_COL = "TRACKER_ID";
const std::string query_manager::TRACKER_TYPE_COL = "TRACKER_TYPE";
const std::string query_manager::TRACK_ATTRS_COL = "TRACK_ATTRS";
const std::string query_manager::TRACK_STATUS_COL = "TRACK_STATUS";

const std::string query_manager::TRACK_DATE_CREATED_COL = "TRACK_DATE_CREATED";
const std::string query_manager::TRACK_DATE_MODIFIED_COL = "TRACK_DATE_MODIFIED";
const std::string query_manager::TRACK_ID_SEQ = "TRACK_TRACK_ID_SEQ";

//pvo_table definitions
const std::string query_manager::PVO_TABLE_NAME = "PVO";
const std::string query_manager::PVO_ID_PK_COL = "PVO_ID";
const std::string query_manager::PVO_TRACK_ID_COL = "PVO_TRACK_ID";
const std::string query_manager::PVO_FRAME_NUM_COL = "PVO_FRAME_NUM";
const std::string query_manager::PVO_PERSON_PROBABILITY_COL = "PERSON_PROBABILITY";
const std::string query_manager::PVO_VEHICLE_PROBABILITY_COL = "VEHICLE_PROBABILITY";
const std::string query_manager::PVO_OTHER_PROBABILITY_COL = "OTHER_PROBABILITY";

//track_state table definitions
const std::string query_manager::STATE_TABLE_NAME = "TRACK_STATE";
const std::string query_manager::STATE_ID_PK_COL = "STATE_ID";
const std::string query_manager::STATE_TRACK_ID_COL = "STATE_TRACK_ID";
const std::string query_manager::STATE_FRAME_NUM_COL = "FRAME_NUM";
const std::string query_manager::STATE_FRAME_TIME_COL = "FRAME_TIME";
const std::string query_manager::STATE_LOC_COL = "STATE_LOC";
const std::string query_manager::STATE_WORLD_LOC_COL = "STATE_WORLD_lOC";
const std::string query_manager::STATE_IMG_LOC_COL = "STATE_IMG_LOC";

const std::string query_manager::STATE_UTM_SMOOTHED_LOC_E_COL = "STATE_UTM_SMOOTHED_LOC_E";
const std::string query_manager::STATE_UTM_SMOOTHED_LOC_N_COL = "STATE_UTM_SMOOTHED_LOC_N";
const std::string query_manager::STATE_UTM_SMOOTHED_LOC_ZONE_COL = "STATE_UTM_SMOOTHED_LOC_ZONE";
const std::string query_manager::STATE_UTM_SMOOTHED_LOC_IS_NORTH_COL = "STATE_UTM_SMOOTHED_LOC_IS_NORTH";
const std::string query_manager::STATE_UTM_RAW_LOC_E_COL = "STATE_UTM_RAW_LOC_E";
const std::string query_manager::STATE_UTM_RAW_LOC_N_COL = "STATE_UTM_RAW_LOC_N";
const std::string query_manager::STATE_UTM_RAW_LOC_ZONE_COL = "STATE_UTM_RAW_LOC_ZONE";
const std::string query_manager::STATE_UTM_RAW_LOC_ZONE_IS_NORTH_COL = "STATE_UTM_RAW_LOC_ZONE_IS_NORTH";
const std::string query_manager::STATE_UTM_VELOCITY_X_COL = "STATE_UTM_VELOCITY_X";
const std::string query_manager::STATE_UTM_VELOCITY_Y_COL = "STATE_UTM_VELOCITY_Y";

const std::string query_manager::STATE_LONLAT_COL = "STATE_LONLAT";
const std::string query_manager::STATE_VELOCITY_X_COL = "STATE_VEL_X";
const std::string query_manager::STATE_VELOCITY_Y_COL = "STATE_VEL_Y";
const std::string query_manager::STATE_BBOX_COL = "STATE_BBOX";
const std::string query_manager::STATE_AREA_COL = "STATE_AREA";
const std::string query_manager::STATE_ATTRS_COL = "STATE_ATTRS";

const std::string query_manager::STATE_IMAGE_CHIP_COL = "IMAGE_CHIP";
const std::string query_manager::STATE_IMAGE_MASK_COL = "IMAGE_MASK";
const std::string query_manager::STATE_IMAGE_CHIP_OFFSET_COL = "IMAGE_CHIP_OFFSET";
const std::string query_manager::STATE_MASK_i0_COL = "MASK_i0";
const std::string query_manager::STATE_MASK_j0_COL = "MASK_j0";
const std::string query_manager::STATE_SESSION_ID_COL = "STATE_SESSION_ID";

const std::string query_manager::STATE_COV_R0C0_COL = "STATE_COV_R0C0";
const std::string query_manager::STATE_COV_R0C1_COL = "STATE_COV_R0C1";
const std::string query_manager::STATE_COV_R1C0_COL = "STATE_COV_R1C0";
const std::string query_manager::STATE_COV_R1C1_COL = "STATE_COV_R1C1";

//event table definitions
const std::string query_manager::EVENT_TABLE_NAME = "EVENT";
const std::string query_manager::EVENT_ID_PK_COL = "EVENT_ID";
const std::string query_manager::EVENT_UUID_COL = "EVENT_UUID";
const std::string query_manager::EVENT_EXT_ID_COL = "EVENT_EXT_ID";
const std::string query_manager::EVENT_SESSION_ID_COL = "EVENT_SESSION_ID";
const std::string query_manager::EVENT_TYPE_COL = "EVENT_TYPE";
const std::string query_manager::EVENT_ST_TIME_COL = "EVENT_ST_TIME";
const std::string query_manager::EVENT_END_TIME_COL = "EVENT_END_TIME";
const std::string query_manager::EVENT_ST_FRAME_COL = "EVENT_ST_FRAME";
const std::string query_manager::EVENT_END_FRAME_COL = "EVENT_END_FRAME";
const std::string query_manager::EVENT_PROBABILITY_COL = "EVENT_PROBABILITY";
const std::string query_manager::EVENT_BB_MIN_X_COL = "EVENT_BB_MIN_X";
const std::string query_manager::EVENT_BB_MIN_Y_COL = "EVENT_BB_MIN_Y";
const std::string query_manager::EVENT_BB_MAX_X_COL = "EVENT_BB_MAX_X";
const std::string query_manager::EVENT_BB_MAX_Y_COL = "EVENT_BB_MAX_Y";
const std::string query_manager::EVENT_DIRECTION_COL = "EVENT_DIRECTION";
const std::string query_manager::EVENT_STATUS_COL = "EVENT_STATUS";
const std::string query_manager::EVENT_DATE_CREATED_COL = "EVENT_DATE_CREATED";
const std::string query_manager::EVENT_DATE_MODIFIED_COL = "EVENT_DATE_MODIFIED";
const std::string query_manager::EVENT_ID_SEQ = "EVENT_EVENT_ID_SEQ";

//event track table definitions
const std::string query_manager::EVENT_TRACK_TABLE_NAME = "EVENT_TRACK";
const std::string query_manager::EVENT_TRACK_ID_PK_COL = "ET_ID";
const std::string query_manager::EVENT_TRACK_EVENT_ID_COL = "ET_EVENT_ID";
const std::string query_manager::EVENT_TRACK_TRACK_ID_COL = "ET_TRACK_ID";
const std::string query_manager::EVENT_TRACK_ST_TIME_COL = "ET_TRACK_ST_TIME";
const std::string query_manager::EVENT_TRACK_END_TIME_COL = "ET_TRACK_END_TIME";
const std::string query_manager::EVENT_TRACK_ST_FRAME_COL = "ET_TRACK_ST_FRAME";
const std::string query_manager::EVENT_TRACK_END_FRAME_COL = "ET_TRACK_END_FRAME";
const std::string query_manager::EVENT_TRACK_POS_COL = "ET_VECTOR_POS";

//activity table definitions
const std::string query_manager::ACT_TABLE_NAME = "ACTIVITY";
const std::string query_manager::ACT_ID_PK_COL = "ACT_ID";
const std::string query_manager::ACT_UUID_COL = "ACT_UUID";
const std::string query_manager::ACT_EXT_ID_COL = "ACT_EXT_ID";
const std::string query_manager::ACT_SESSION_ID_COL = "ACT_SESSION_ID";
const std::string query_manager::ACT_TYPE_COL = "ACT_TYPE";
const std::string query_manager::ACT_PROBABILITY_COL = "ACT_PROBABILITY";
const std::string query_manager::ACT_NORMALCY_COL = "ACT_NORMALCY";
const std::string query_manager::ACT_SALIENCY_COL = "ACT_SALIENCY";
const std::string query_manager::ACT_ST_TIME_COL = "ACT_ST_TIME";
const std::string query_manager::ACT_ST_FRAME_COL = "ACT_ST_FRAME";
const std::string query_manager::ACT_END_TIME_COL = "ACT_END_TIME";
const std::string query_manager::ACT_END_FRAME_COL = "ACT_END_FRAME";
const std::string query_manager::ACT_BB_MIN_X_COL = "ACT_BB_MIN_X";
const std::string query_manager::ACT_BB_MIN_Y_COL = "ACT_BB_MIN_Y";
const std::string query_manager::ACT_BB_MAX_X_COL = "ACT_BB_MAX_X";
const std::string query_manager::ACT_BB_MAX_Y_COL = "ACT_BB_MAX_Y";
const std::string query_manager::ACT_DATE_CREATED_COL = "ACT_DATE_CREATED";
const std::string query_manager::ACT_DATE_MODIFIED_COL = "ACT_DATE_MODIFIED";

const std::string query_manager::ACT_ID_SEQ = "ACTIVITY_ACT_ID_SEQ";

//activity event table definitions
const std::string query_manager::ACT_EVENT_TABLE_NAME = "ACTIVITY_EVENT";
const std::string query_manager::ACT_EVENT_ACT_ID_COL = "AE_ACT_ID";
const std::string query_manager::ACT_EVENT_EVENT_ID_COL = "AE_EVENT_ID";
const std::string query_manager::ACT_EVENT_POS_COL = "AE_VECTOR_POS";

//frame_metadata definitions
const std::string query_manager::VIDTK_FRAME_METADATA_TABLE_NAME = "VIDTK_FRAME_METADATA";
const std::string query_manager::VFM_TS_ID_COL = "VFM_TS_ID";
const std::string query_manager::VFM_MISSION_ID_COL = "VFM_MISSION_ID";
const std::string query_manager::VFM_FRAME_NUM_COL = "VFM_FRAME_NUM";
const std::string query_manager::VFM_FRAME_TIME_COL = "VFM_FRAME_TIME";
const std::string query_manager::VFM_FILE_PATH_COL = "VFM_FILE_PATH";

const std::string query_manager::VIDTK_FRAME_PRODUCTS_TABLE_NAME = "VIDTK_FRAME_PRODUCTS";
const std::string query_manager::VFP_ID_COL = "VFP_ID";
const std::string query_manager::VFP_SESSION_ID_COL = "VFP_SESSION_ID";
const std::string query_manager::VFP_VIDTK_TS_COL = "VFP_VIDTK_TS";
const std::string query_manager::VFP_SRC_TO_REF_REF_TS_COL = "VFP_SRC_TO_REF_REF_TS";
const std::string query_manager::VFP_SRC_TO_REF_SRC_TS_COL = "VFP_SRC_TO_REF_SRC_TS";
const std::string query_manager::VFP_SRC_TO_REF_H_R0C0_COL = "VFP_SRC_TO_REF_H_R0C0";
const std::string query_manager::VFP_SRC_TO_REF_H_R0C1_COL = "VFP_SRC_TO_REF_H_R0C1";
const std::string query_manager::VFP_SRC_TO_REF_H_R0C2_COL = "VFP_SRC_TO_REF_H_R0C2";
const std::string query_manager::VFP_SRC_TO_REF_H_R1C0_COL = "VFP_SRC_TO_REF_H_R1C0";
const std::string query_manager::VFP_SRC_TO_REF_H_R1C1_COL = "VFP_SRC_TO_REF_H_R1C1";
const std::string query_manager::VFP_SRC_TO_REF_H_R1C2_COL = "VFP_SRC_TO_REF_H_R1C2";
const std::string query_manager::VFP_SRC_TO_REF_H_R2C0_COL = "VFP_SRC_TO_REF_H_R2C0";
const std::string query_manager::VFP_SRC_TO_REF_H_R2C1_COL = "VFP_SRC_TO_REF_H_R2C1";
const std::string query_manager::VFP_SRC_TO_REF_H_R2C2_COL = "VFP_SRC_TO_REF_H_R2C2";
const std::string query_manager::VFP_SRC_TO_REF_IS_VALID_COL = "VFP_SRC_TO_REF_IS_VALID";
const std::string query_manager::VFP_SRC_TO_REF_IS_NEW_REF_COL = "VFP_SRC_TO_REF_IS_NEW_REF";
const std::string query_manager::VFP_SRC_TO_UTM_ZONE_COL = "VFP_SRC_TO_UTM_ZONE";
const std::string query_manager::VFP_SRC_TO_UTM_NORTHING_COL = "VFP_SRC_TO_UTM_NORTHING";
const std::string query_manager::VFP_SRC_TO_UTM_SRC_TS_COL = "VFP_SRC_TO_UTM_SRC_TS";
const std::string query_manager::VFP_SRC_TO_UTM_H_R0C0_COL = "VFP_SRC_TO_UTM_H_R0C0";
const std::string query_manager::VFP_SRC_TO_UTM_H_R0C1_COL = "VFP_SRC_TO_UTM_H_R0C1";
const std::string query_manager::VFP_SRC_TO_UTM_H_R0C2_COL = "VFP_SRC_TO_UTM_H_R0C2";
const std::string query_manager::VFP_SRC_TO_UTM_H_R1C0_COL = "VFP_SRC_TO_UTM_H_R1C0";
const std::string query_manager::VFP_SRC_TO_UTM_H_R1C1_COL = "VFP_SRC_TO_UTM_H_R1C1";
const std::string query_manager::VFP_SRC_TO_UTM_H_R1C2_COL = "VFP_SRC_TO_UTM_H_R1C2";
const std::string query_manager::VFP_SRC_TO_UTM_H_R2C0_COL = "VFP_SRC_TO_UTM_H_R2C0";
const std::string query_manager::VFP_SRC_TO_UTM_H_R2C1_COL = "VFP_SRC_TO_UTM_H_R2C1";
const std::string query_manager::VFP_SRC_TO_UTM_H_R2C2_COL = "VFP_SRC_TO_UTM_H_R2C2";
const std::string query_manager::VFP_SRC_TO_UTM_IS_VALID_COL = "VFP_SRC_TO_UTM_IS_VALID";
const std::string query_manager::VFP_SRC_TO_UTM_IS_NEW_COL = "VFP_SRC_TO_UTM_IS_NEW";
const std::string query_manager::VFP_COMPUTED_GSD_COL = "VFP_COMPUTED_GSD";

const std::string query_manager::VIDTK_TILE_METADATA_TABLE_NAME = "VIDTK_TILE_METADATA";
const std::string query_manager::VTM_ID_COL = "VTM_ID";
const std::string query_manager::VTM_SESSION_ID_COL = "VTM_SESSION_ID";
const std::string query_manager::VTM_VIDTK_TS_COL = "VTM_VIDTK_TS";
const std::string query_manager::VTM_UPPER_LEFT_LAT_COL = "VTM_UPPER_LEFT_LAT";
const std::string query_manager::VTM_UPPER_LEFT_LON_COL = "VTM_UPPER_LEFT_LON";
const std::string query_manager::VTM_UPPER_RIGHT_LAT_COL = "VTM_UPPER_RIGHT_LAT";
const std::string query_manager::VTM_UPPER_RIGHT_LON_COL = "VTM_UPPER_RIGHT_LON";
const std::string query_manager::VTM_LOWER_RIGHT_LAT_COL = "VTM_LOWER_RIGHT_LAT";
const std::string query_manager::VTM_LOWER_RIGHT_LON_COL = "VTM_LOWER_RIGHT_LON";
const std::string query_manager::VTM_LOWER_LEFT_LAT_COL = "VTM_LOWER_LEFT_LAT";
const std::string query_manager::VTM_LOWER_LEFT_LON_COL = "VTM_LOWER_LEFT_LON";
const std::string query_manager::VTM_UPPER_LEFT_X_OFFSET_COL = "VTM_UPPER_LEFT_X_OFFSET";
const std::string query_manager::VTM_UPPER_LEFT_Y_OFFSET_COL = "VTM_UPPER_LEFT_Y_OFFSET";
const std::string query_manager::VTM_PIXEL_WIDTH_COL = "VTM_PIXEL_WIDTH";
const std::string query_manager::VTM_PIXEL_HEIGHT_COL = "VTM_PIXEL_HEIGHT";

query_manager
::query_manager()
{

}

query_manager
::~query_manager()
{

}

const std::string &
query_manager
::insert_pvo_query() const
{
  static std::string INSERT_PVO_SQL =
    "INSERT INTO " + PVO_TABLE_NAME       +
    " ( " + PVO_TRACK_ID_COL              +
    " , " + PVO_FRAME_NUM_COL             +
    " , " + PVO_PERSON_PROBABILITY_COL    +
    " , " + PVO_VEHICLE_PROBABILITY_COL   +
    " , " + PVO_OTHER_PROBABILITY_COL     +
    " ) VALUES (?,?,?,?,?)";

  return INSERT_PVO_SQL;
}

const std::string &
query_manager
::select_track_by_id_query() const
{
  static std::string SELECT_TRACK_BY_ID_SQL =
    "SELECT "
    + get_track_column_select() + " , "
    + get_pvo_column_select() + " , "
    + get_track_state_column_select()
    + " FROM " + get_track_table_select()
    + " WHERE " + TRACK_ID_PK_COL + " = ?"
    + " ORDER BY " +  STATE_FRAME_NUM_COL;

  return SELECT_TRACK_BY_ID_SQL;
}


const std::string &
query_manager
::select_all_tracks_query() const
{
  ///Queries that select a large group end abruptly to facilitate a dynamic in clause
  static std::string SELECT_ALL_TRACKS_SQL =
    "SELECT "
    + get_track_column_select() + " , "
    + get_pvo_column_select() + " , "
    + get_track_state_column_select()
    + " FROM " + get_track_table_select()
    + " WHERE " + STATE_SESSION_ID_COL + " =? "
    + " AND " + TRACK_ID_PK_COL;


  return SELECT_ALL_TRACKS_SQL;
}


const std::string &
query_manager
::select_all_tracks_by_time_query() const
{
  static std::string SELECT_TRACKS_BY_TIME =
    "SELECT "
    + get_track_column_select() + " , "
    + get_pvo_column_select() + " , "
    + get_track_state_column_select()
    + " FROM " + get_track_table_select()

    + " WHERE " + TRACK_START_TIME_COL  + " >= ?"
    + " AND "   + TRACK_END_TIME_COL    + " <= ? "
    + " AND "   + STATE_SESSION_ID_COL  + " =? "
    + " ORDER BY " + TRACK_EXT_ID_COL
    + " , " + STATE_FRAME_NUM_COL;

  return SELECT_TRACKS_BY_TIME;
}


const std::string &
query_manager
::select_all_tracks_by_area_query() const
{
  /*
    SQL gets all track_ids in a geo-spatial window
  */
  static std::string SELECT_ALL_TRACK_IDS_BY_AREA =
    "SELECT DISTINCT (" + TRACK_ID_PK_COL          +
    " ) FROM " + TRACK_TABLE_NAME +
    " WHERE ( ST_Intersects("   + TRACK_REGION_COL +
    "::geometry, " + POLYGON +
    "::geometry )  AND " + STATE_SESSION_ID_COL   +
    " =? )";

  return SELECT_ALL_TRACK_IDS_BY_AREA;
}

const std::string &
query_manager
::select_all_tracks_by_area_and_time_query() const
{
  /*
    SQL gets all track_ids in a geo-spatial/temporal window
  */
  static std::string SELECT_ALL_TRACK_IDS_BY_AREA_AND_TIME =
    "SELECT DISTINCT ( " + TRACK_ID_PK_COL          +
    " ) FROM " + TRACK_TABLE_NAME +
    " WHERE ( ST_Intersects("   + TRACK_REGION_COL +
    "::geometry, " + POLYGON +
    "::geometry ) AND ( "  + TRACK_START_TIME_COL +
    " >= ?   AND   "  + TRACK_END_TIME_COL +
    " <= ? ) AND ( " + TRACK_SESSION_ID_COL   +
    " =?))";

  return SELECT_ALL_TRACK_IDS_BY_AREA_AND_TIME;
}

const std::string &
query_manager
::select_track_by_uuid_query() const
{
  static std::string SELECT_TRACK_BY_UUID_SQL =
    "SELECT "

    + get_track_column_select() + " , "
    + get_pvo_column_select() + " , "
    + get_track_state_column_select()
    + " FROM " + get_track_table_select()

    + " WHERE " + TRACK_UUID_COL +  " = ?"
    + " ORDER BY " +  STATE_FRAME_NUM_COL;

  return SELECT_TRACK_BY_UUID_SQL;
}

const std::string &
query_manager
::select_all_track_ids_query() const
{
  static std::string SELECT_ALL_TRACK_IDS =
    " SELECT " + TRACK_ID_PK_COL +
    " FROM   " + TRACK_TABLE_NAME +
    " WHERE " + TRACK_SESSION_ID_COL +
    " =? "    +                      +
    " ORDER BY " + TRACK_EXT_ID_COL;

  return SELECT_ALL_TRACK_IDS;
}

const std::string &
query_manager
::select_max_track_ext_id_query() const
{
  static std::string SELECT_MAX_EXT_ID =
    "SELECT MAX( "       + TRACK_EXT_ID_COL     +
    " ) AS max_id FROM " + TRACK_TABLE_NAME     +
    " WHERE "            + TRACK_SESSION_ID_COL +
    " =? ";

  return SELECT_MAX_EXT_ID;
}


const std::string &
query_manager
::select_max_frame_number_query() const
{
  static std::string SELECT_MAX_FRAME_NUMBER =
    "SELECT MAX( "           + STATE_FRAME_NUM_COL +
    " ) as maxframe FROM "   + TRACK_TABLE_NAME    +
    " INNER JOIN "   + STATE_TABLE_NAME     +
    " ON ( "         + STATE_TRACK_ID_COL   +
    " = "            + TRACK_ID_PK_COL      + " ) " +
    " WHERE "      + STATE_SESSION_ID_COL +
    " = ?";

  return SELECT_MAX_FRAME_NUMBER;
}


const std::string &
query_manager
::new_select_trackids_by_last_frame_query() const
{
  static std::string NEW_SELECT_TRACKIDS_BY_LAST_FRAME =
    "SELECT "        + TRACK_ID_PK_COL   +
    " ,MAX ( "       + STATE_FRAME_NUM_COL  +
    " ) as maxframe FROM "      + TRACK_TABLE_NAME     +
    " INNER JOIN "   + STATE_TABLE_NAME     +
    " ON ( "         + STATE_TRACK_ID_COL   +
    " = "            + TRACK_ID_PK_COL      +
    " ) WHERE "      + STATE_SESSION_ID_COL +
    " = ? GROUP BY( " + TRACK_ID_PK_COL     +
    ")";

  return NEW_SELECT_TRACKIDS_BY_LAST_FRAME;
}


const std::string &
query_manager
::select_trackids_by_last_frame_query() const
{
  static std::string SELECT_TRACKIDS_BY_LAST_FRAME =
    "SELECT "        + TRACK_ID_PK_COL   +
    " ,MAX ( "       + STATE_FRAME_NUM_COL  +
    " )  FROM "      + TRACK_TABLE_NAME     +
    " INNER JOIN "   + STATE_TABLE_NAME     +
    " ON ( "         + STATE_TRACK_ID_COL   +
    " = "            + TRACK_ID_PK_COL      +
    " ) WHERE "      + STATE_SESSION_ID_COL +
    " = ? GROUP BY( " + TRACK_ID_PK_COL     +
    ") HAVING MAX ( " + STATE_FRAME_NUM_COL +
    " ) = ?";

  return SELECT_TRACKIDS_BY_LAST_FRAME;
}


const std::string &
query_manager
::select_tracks_by_frame_range_query() const
{
  static std::string SELECT_TRACKS_BY_FRAME_RANGE =
    "SELECT "

    + get_track_column_select() + " , "
    + get_pvo_column_select() + " , "
    + get_track_state_column_select()
    + " FROM " + get_track_table_select()


    + " WHERE " + STATE_SESSION_ID_COL + " =? "
    + " AND " + STATE_FRAME_NUM_COL + " >=? "
    + " AND " + STATE_FRAME_NUM_COL + " <=? "
    + " ORDER BY " + TRACK_ID_PK_COL +
    + " , " + STATE_FRAME_NUM_COL;

  return SELECT_TRACKS_BY_FRAME_RANGE;
}

const std::string
query_manager
::select_tracks_by_frame_range_multi_session_query( int size )
{
  static std::string SELECT_TRACKS_BY_FRAME_RANGE_HEAD =
    "SELECT "

    + get_track_column_select() + " , "
    + get_pvo_column_select() + " , "
    + get_track_state_column_select()
    + " FROM " + get_track_table_select()


    + " WHERE " + STATE_SESSION_ID_COL + "  ";

    std::string in_clause = make_in_clause( size );

    std::string SELECT_TRACKS_BY_FRAME_RANGE_TAIL = " AND " + STATE_FRAME_NUM_COL + " >=? "
    + " AND " + STATE_FRAME_NUM_COL + " <=? "
    + " ORDER BY " + TRACK_ID_PK_COL +
    + " , " + STATE_FRAME_NUM_COL;

  return SELECT_TRACKS_BY_FRAME_RANGE_HEAD + in_clause + SELECT_TRACKS_BY_FRAME_RANGE_TAIL;
}


const std::string &
query_manager
::select_active_tracks_by_frame_number_query() const
{
  /*
    NOTE, Below is the translation of the query below, provided
          for convenience and testing.  Anyone who changes the
          underlying query must make sure to change this translation as well.


    SELECT
       TRACK_ID , TRACK_UUID , TRACK_EXT_ID , LAST_MOD_MATCH , FALSE_ALARM ,
       TRACKER_ID , TRACKER_TYPE ,  TRACK_ATTRS , PERSON_PROBABILITY ,
       VEHICLE_PROBABILITY , OTHER_PROBABILITY , FRAME_NUM , FRAME_TIME , Astext( STATE_LOC ),
       AsText( STATE_WORLD_lOC )  , AsText( STATE_IMG_LOC )  , AsText( STATE_LONLAT ),
       STATE_VEL_X , STATE_VEL_Y ,  AsText( STATE_BBOX )  , STATE_ATTRS , STATE_AREA ,
       IMAGE_CHIP , IMAGE_MASK , IMAGE_CHIP_OFFSET , MASK_i0 , MASK_j0
    FROM TRACK
    LEFT OUTER JOIN PVO ON
       ( PVO_TRACK_ID = TRACK_ID and pvo_frame_num =
       (select max ( pvo_frame_num) from pvo where  PVO_TRACK_ID = TRACK_ID) )
    INNER JOIN TRACK_STATE ON ( TRACK_ID = STATE_TRACK_ID )
    WHERE TRACK_ID IN
    (
       SELECT DISTINCT( TRACK_ID )
       FROM TRACK
       INNER JOIN TRACK_STATE ON ( TRACK_ID = STATE_TRACK_ID )
       WHERE TRACK_SESSION_ID =?
       AND ( FRAME_NUM <=?  AND TRACK_STATUS IS NULL  OR TRACK_STATUS >? )
    )
    AND FRAME_NUM <=?  ORDER BY TRACK_ID , FRAME_NUM
   */
  static std::string SELECT_ACTIVE_TRACKS_BY_FRAME_NUMBER =
    "SELECT "
    + get_track_column_select() + " , "
    + get_pvo_column_select() + " , "
    + get_track_state_column_select()
    + " FROM " + get_track_table_select() +


    " WHERE " + TRACK_ID_PK_COL +
    " IN ( SELECT DISTINCT( " + TRACK_ID_PK_COL + " ) " +
    " FROM " + TRACK_TABLE_NAME +
    " INNER JOIN " + STATE_TABLE_NAME +
    " ON ( " + TRACK_ID_PK_COL +
    " = " + STATE_TRACK_ID_COL +  " ) " +
    " WHERE " + STATE_SESSION_ID_COL + " =? " +
    " AND ( " + STATE_FRAME_NUM_COL + " <=? " +
    " AND " + TRACK_STATUS_COL + " IS NULL " +
    " OR " + TRACK_STATUS_COL + " >? ) ) " +
    " AND " + STATE_FRAME_NUM_COL + " <=? " +
    " ORDER BY " + TRACK_ID_PK_COL +
    " , " + STATE_FRAME_NUM_COL;

  return SELECT_ACTIVE_TRACKS_BY_FRAME_NUMBER;
}


/*
    NOTE, Below is the translation of the query below, provided
          for convenience and testing.  Anyone who changes the
          underlying query must make sure to change this translation as well.

    SELECT
       TRACK_ID , TRACK_SESSION_ID , TRACK_UUID , TRACK_EXT_ID , LAST_MOD_MATCH , FALSE_ALARM ,
       TRACKER_ID , TRACKER_TYPE , TRACK_ATTRS , PERSON_PROBABILITY , VEHICLE_PROBABILITY ,
       OTHER_PROBABILITY , FRAME_NUM , FRAME_TIME ,asText( STATE_LOC )  ,asText( STATE_WORLD_lOC )  ,
       asText( STATE_IMG_LOC )  ,asText( STATE_LONLAT )  , STATE_VEL_X , STATE_VEL_Y ,asText( STATE_BBOX )  ,
       STATE_ATTRS , STATE_AREA , IMAGE_CHIP , IMAGE_MASK , IMAGE_CHIP_OFFSET , MASK_i0 , MASK_j0
    FROM TRACK
    LEFT OUTER JOIN PVO ON
       ( PVO_TRACK_ID = TRACK_ID and pvo_frame_num =
       (select max ( pvo_frame_num) from pvo where  PVO_TRACK_ID = TRACK_ID)   )
    INNER JOIN TRACK_STATE ON ( STATE_TRACK_ID = TRACK_ID )
    WHERE TRACK_ID IN
    (
       SELECT DISTINCT( TRACK_ID )
       FROM TRACK INNER JOIN TRACK_STATE ON ( TRACK_ID = STATE_TRACK_ID )
       WHERE TRACK_SESSION_ID  in ( list_of_session_ids )
       AND ( FRAME_NUM <=?  AND TRACK_STATUS IS NULL  OR TRACK_STATUS >? )
    )
    AND FRAME_NUM <=?  ORDER BY TRACK_ID , FRAME_NUM

*/
const std::string
query_manager
::select_active_tracks_by_frame_number_multi_session_query( int size )
{
  static std::string SELECT_ACTIVE_TRACKS_BY_FRAME_NUMBER_MULTI_HEAD =
    "SELECT "
    + get_track_column_select() + " , "
    + get_pvo_column_select() + " , "
    + get_track_state_column_select()
    + " FROM " + get_track_table_select() +

    " WHERE " + TRACK_ID_PK_COL +
    " IN ( SELECT DISTINCT( " + TRACK_ID_PK_COL + " ) " +
    " FROM " + TRACK_TABLE_NAME +
    " INNER JOIN " + STATE_TABLE_NAME +
    " ON ( " + TRACK_ID_PK_COL +
    " = " + STATE_TRACK_ID_COL +  " ) " +
    " WHERE " + STATE_SESSION_ID_COL + " ";

  std::string in_clause = make_in_clause( size );

  static std::string SELECT_ACTIVE_TRACKS_BY_FRAME_NUMBER_MULTI_TAIL =
    " AND ( " + STATE_FRAME_NUM_COL + " <=? " +
    " AND " + TRACK_STATUS_COL + " IS NULL " +
    " OR " + TRACK_STATUS_COL + " >? ) ) " +
    " AND " + STATE_FRAME_NUM_COL + " <=? " +
    " ORDER BY " + TRACK_ID_PK_COL +
    " , " + STATE_FRAME_NUM_COL;

  return
    SELECT_ACTIVE_TRACKS_BY_FRAME_NUMBER_MULTI_HEAD +
    in_clause +
    SELECT_ACTIVE_TRACKS_BY_FRAME_NUMBER_MULTI_TAIL;

}


const std::string &
query_manager
::select_active_tracks_with_exclusion_session_query() const
{
  /*
    NOTE, Below is the translation of the query below, provided
          for convenience and testing.  Anyone who changes the
          underlying query must make sure to change this translation as well.

    SELECT
       TRACK_ID , TRACK_UUID , TRACK_EXT_ID , LAST_MOD_MATCH , FALSE_ALARM ,
       TRACKER_ID , TRACKER_TYPE , TRACK_ATTRS , PERSON_PROBABILITY ,
       VEHICLE_PROBABILITY , OTHER_PROBABILITY , FRAME_NUM , FRAME_TIME , Astext( STATE_LOC ),
       AsText( STATE_WORLD_lOC )  , AsText( STATE_IMG_LOC )  , AsText( STATE_LONLAT ),
       STATE_VEL_X , STATE_VEL_Y ,  AsText( STATE_BBOX )  , STATE_ATTRS , STATE_AREA ,
       IMAGE_CHIP , IMAGE_MASK , IMAGE_CHIP_OFFSET , MASK_i0 , MASK_j0
    FROM TRACK LEFT OUTER JOIN PVO ON ( PVO_TRACK_ID = TRACK_ID  )
    INNER JOIN TRACK_STATE ON ( TRACK_ID = STATE_TRACK_ID )
    WHERE TRACK_ID IN
    (
       SELECT DISTINCT( TRACK_ID )
       FROM TRACK
       INNER JOIN TRACK_STATE ON ( TRACK_ID = STATE_TRACK_ID )
       WHERE TRACK_SESSION_ID =?
       AND ( FRAME_NUM <=?  AND TRACK_STATUS IS NULL  OR TRACK_STATUS >? )
    )
    AND FRAME_NUM <=?
    AND TRACK_UUID
    NOT IN
    (
       SELECT DISTINCT( TRACK_uuID )
       FROM TRACK
       WHERE TRACK_SESSION_ID = ?
       AND TRACK_STATUS  <= ?
    )
  ORDER BY TRACK_ID , FRAME_NUM
   */
  static std::string SELECT_ACTIVE_TRACKS_WITH_EXCLUSION_SESSION =
    "SELECT "

    + get_track_column_select() + " , "
    + get_pvo_column_select() + " , "
    + get_track_state_column_select()
    + " FROM " + get_track_table_select() +

    " WHERE " + TRACK_ID_PK_COL +
    " IN ( SELECT DISTINCT( " + TRACK_ID_PK_COL + " ) " +
    " FROM " + TRACK_TABLE_NAME +
    " INNER JOIN " + STATE_TABLE_NAME +
    " ON ( " + TRACK_ID_PK_COL +
    " = " + STATE_TRACK_ID_COL +  " ) " +
    " WHERE " + TRACK_SESSION_ID_COL + " =? " +
    " AND ( " + STATE_FRAME_NUM_COL + " <=? " +
    " AND " + TRACK_STATUS_COL + " IS NULL " +
    " OR " + TRACK_STATUS_COL + " >? ) ) " +
    " AND " + STATE_FRAME_NUM_COL + " <=? " +
    " AND TRACK_UUID NOT IN ( " +
      "  SELECT DISTINCT( TRACK_uuID ) " +
       " FROM TRACK" +
       " WHERE TRACK_SESSION_ID = ? " +
       " AND TRACK_STATUS  <= ?  )" +
    " ORDER BY " + TRACK_ID_PK_COL +
    " , " + STATE_FRAME_NUM_COL;

  return SELECT_ACTIVE_TRACKS_WITH_EXCLUSION_SESSION;
}


const std::string &
query_manager
::insert_event_query() const
{
  static std::string INSERT_EVENT_SQL =
    "INSERT INTO " + EVENT_TABLE_NAME   +
    " ( " + EVENT_UUID_COL             +
    " , " + EVENT_EXT_ID_COL            +
    " , " + EVENT_SESSION_ID_COL        +
    " , " + EVENT_TYPE_COL              +
    " , " + EVENT_ST_TIME_COL           +
    " , " + EVENT_ST_FRAME_COL          +
    " , " + EVENT_END_TIME_COL          +
    " , " + EVENT_END_FRAME_COL         +
    " , " + EVENT_PROBABILITY_COL       +
    " , " + EVENT_BB_MIN_X_COL  +
    " , " + EVENT_BB_MIN_Y_COL  +
    " , " + EVENT_BB_MAX_X_COL  +
    " , " + EVENT_BB_MAX_Y_COL  +
    " , " + EVENT_DIRECTION_COL +
    " , " + EVENT_STATUS_COL    +
    " , " + EVENT_DATE_CREATED_COL +
    " ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)";

  return INSERT_EVENT_SQL;
}

const std::string &
query_manager
::insert_event_track_query() const
{
  static std::string INSERT_EVENT_TRACK_SQL =
    "INSERT INTO " + EVENT_TRACK_TABLE_NAME   +
    " ( " + EVENT_TRACK_EVENT_ID_COL          +
    " , " + EVENT_TRACK_TRACK_ID_COL          +
    " , " + EVENT_TRACK_ST_TIME_COL           +
    " , " + EVENT_TRACK_ST_FRAME_COL          +
    " , " + EVENT_TRACK_END_TIME_COL          +
    " , " + EVENT_TRACK_END_FRAME_COL         +
    " , " + EVENT_TRACK_POS_COL               +
    " ) VALUES (?,?,?,?,?,?,?)";

  return INSERT_EVENT_TRACK_SQL;
}

const std::string &
query_manager
::track_uuid_by_id_query() const
{
  static std::string TRACK_UUID_BY_ID =
    "SELECT " + TRACK_UUID_COL   +
    " , "     + TRACK_EXT_ID_COL +
    " FROM  " + TRACK_TABLE_NAME +
    " WHERE " + TRACK_ID_PK_COL  +
    " =?";

  return TRACK_UUID_BY_ID;
}

const std::string &
query_manager
::track_id_by_uuid_query() const
{
  static std::string TRACK_ID_BY_UUID =
    "SELECT " + TRACK_ID_PK_COL  +
    " FROM  " + TRACK_TABLE_NAME +
    " WHERE " + TRACK_UUID_COL   +
    " =? AND " + TRACK_SESSION_ID_COL +
    " =?";

  return TRACK_ID_BY_UUID;
}

const std::string &
query_manager
::select_event_by_id_query() const
{
  static std::string SELECT_EVENT_BY_ID_SQL =
    "SELECT " + EVENT_UUID_COL            +
    " , "     + EVENT_EXT_ID_COL           +
    " , "     + EVENT_TYPE_COL             +
    " , "     + EVENT_ST_TIME_COL +
    " , "     + EVENT_ST_FRAME_COL +
    " , "     + EVENT_END_TIME_COL +
    " , "     + EVENT_END_FRAME_COL +
    " , "     + EVENT_PROBABILITY_COL      +
    " , "     + EVENT_BB_MIN_X_COL +
    " , "     + EVENT_BB_MIN_Y_COL +
    " , "     + EVENT_BB_MAX_X_COL +
    " , "     + EVENT_BB_MAX_Y_COL +
    " , "     + EVENT_DIRECTION_COL +
    " , "     + EVENT_DATE_CREATED_COL +
    " , "     + EVENT_DATE_MODIFIED_COL +

    " , "     + EVENT_TRACK_TRACK_ID_COL +
    " , "     + EVENT_TRACK_ST_TIME_COL +
    " , "     + EVENT_TRACK_ST_FRAME_COL +
    " , "     + EVENT_TRACK_END_TIME_COL +
    " , "     + EVENT_TRACK_END_FRAME_COL +
    " , "     + EVENT_TRACK_POS_COL +

    " FROM " + EVENT_TABLE_NAME +
    " INNER JOIN " + EVENT_TRACK_TABLE_NAME +
    " ON ( " + EVENT_ID_PK_COL  +
    " = " + EVENT_TRACK_EVENT_ID_COL   +
    " ) WHERE " + EVENT_ID_PK_COL +
    " =? "  +
    " order by " + EVENT_ST_TIME_COL +
    " , " + EVENT_TRACK_POS_COL;

  return SELECT_EVENT_BY_ID_SQL;
}


const std::string &
query_manager
::select_event_by_uuid_query() const
{
  static std::string SELECT_EVENT_BY_UUID_SQL =
    "SELECT " + EVENT_UUID_COL            +
    " , "     + EVENT_EXT_ID_COL           +
    " , "     + EVENT_TYPE_COL             +
    " , "     + EVENT_ST_TIME_COL +
    " , "     + EVENT_ST_FRAME_COL +
    " , "     + EVENT_END_TIME_COL +
    " , "     + EVENT_END_FRAME_COL +
    " , "     + EVENT_PROBABILITY_COL      +
    " , "     + EVENT_BB_MIN_X_COL +
    " , "     + EVENT_BB_MIN_Y_COL +
    " , "     + EVENT_BB_MAX_X_COL +
    " , "     + EVENT_BB_MAX_Y_COL +
    " , "     + EVENT_DIRECTION_COL +
    " , "     + EVENT_DATE_CREATED_COL +
    " , "     + EVENT_DATE_MODIFIED_COL +

    " , "     + EVENT_TRACK_TRACK_ID_COL +
    " , "     + EVENT_TRACK_ST_TIME_COL +
    " , "     + EVENT_TRACK_ST_FRAME_COL +
    " , "     + EVENT_TRACK_END_TIME_COL +
    " , "     + EVENT_TRACK_END_FRAME_COL +
    " , "     + EVENT_TRACK_POS_COL +

    " FROM " + EVENT_TABLE_NAME +
    " INNER JOIN " + EVENT_TRACK_TABLE_NAME +
    " ON ( " + EVENT_ID_PK_COL  +
    " = " + EVENT_TRACK_EVENT_ID_COL   +
    " ) WHERE " + EVENT_UUID_COL +
    " =? "  +
    " order by " + EVENT_ST_TIME_COL +
    " , " + EVENT_TRACK_POS_COL;

  return SELECT_EVENT_BY_UUID_SQL;
}


const std::string &
query_manager
::select_terminated_events_query() const
{
  //*******ALERT*********
  // This query orders data by event_ext_id and secondarily by vector position,
  // if you change that, you MUST verify that the processing in event_db::get_all_events
  // functions properly when adding tracks and start and end times to the event.
  static std::string SELECT_ALL_EVENTS_SQL =
    "SELECT " + EVENT_ID_PK_COL            +
    " , "     + EVENT_UUID_COL             +
    " , "     + EVENT_EXT_ID_COL           +
    " , "     + EVENT_TYPE_COL             +
    " , "     + EVENT_ST_TIME_COL +
    " , "     + EVENT_ST_FRAME_COL +
    " , "     + EVENT_END_TIME_COL +
    " , "     + EVENT_END_FRAME_COL +
    " , "     + EVENT_PROBABILITY_COL      +
    " , "     + EVENT_BB_MIN_X_COL +
    " , "     + EVENT_BB_MIN_Y_COL +
    " , "     + EVENT_BB_MAX_X_COL +
    " , "     + EVENT_BB_MAX_Y_COL +
    " , "     + EVENT_DIRECTION_COL +
    " , "     + EVENT_DATE_CREATED_COL +
    " , "     + EVENT_DATE_MODIFIED_COL +

    " , "     + EVENT_TRACK_TRACK_ID_COL +
    " , "     + EVENT_TRACK_ST_TIME_COL +
    " , "     + EVENT_TRACK_ST_FRAME_COL +
    " , "     + EVENT_TRACK_END_TIME_COL +
    " , "     + EVENT_TRACK_END_FRAME_COL +
    " , "     + EVENT_TRACK_POS_COL +

    " FROM " + EVENT_TABLE_NAME +
    " INNER JOIN " + EVENT_TRACK_TABLE_NAME +
    " ON ( " + EVENT_ID_PK_COL  +
    " = " + EVENT_TRACK_EVENT_ID_COL   +
    " ) WHERE " + EVENT_SESSION_ID_COL +
    " =? AND " + EVENT_STATUS_COL      +
    " =? "     +
    "ORDER BY " + EVENT_ID_PK_COL  +
    ", " + EVENT_TRACK_POS_COL;

  return SELECT_ALL_EVENTS_SQL;
}



const std::string &
query_manager
::select_all_events_query() const
{

  ///Queries that select a large group end abruptly to facilitate a dynamic in clause
  //*******ALERT*********
  // This query orders data by event_ext_id and secondarily by vector position,
  // if you change that, you MUST verify that the processing in event_db::get_all_events
  // functions properly when adding tracks and start and end times to the event.
  static std::string SELECT_ALL_EVENTS_SQL =
    "SELECT " + EVENT_ID_PK_COL            +
    " , "     + EVENT_UUID_COL             +
    " , "     + EVENT_EXT_ID_COL           +
    " , "     + EVENT_TYPE_COL             +
    " , "     + EVENT_ST_TIME_COL +
    " , "     + EVENT_ST_FRAME_COL +
    " , "     + EVENT_END_TIME_COL +
    " , "     + EVENT_END_FRAME_COL +
    " , "     + EVENT_PROBABILITY_COL      +
    " , "     + EVENT_BB_MIN_X_COL +
    " , "     + EVENT_BB_MIN_Y_COL +
    " , "     + EVENT_BB_MAX_X_COL +
    " , "     + EVENT_BB_MAX_Y_COL +
    " , "     + EVENT_DIRECTION_COL +
    " , "     + EVENT_DATE_CREATED_COL +
    " , "     + EVENT_DATE_MODIFIED_COL +

    " , "     + EVENT_TRACK_TRACK_ID_COL +
    " , "     + EVENT_TRACK_ST_TIME_COL +
    " , "     + EVENT_TRACK_ST_FRAME_COL +
    " , "     + EVENT_TRACK_END_TIME_COL +
    " , "     + EVENT_TRACK_END_FRAME_COL +
    " , "     + EVENT_TRACK_POS_COL +

    " FROM " + EVENT_TABLE_NAME +
    " INNER JOIN " + EVENT_TRACK_TABLE_NAME +
    " ON ( " + EVENT_ID_PK_COL  +
    " = " + EVENT_TRACK_EVENT_ID_COL   +
    " ) WHERE " + EVENT_SESSION_ID_COL +
    " =? ORDER BY " + EVENT_ID_PK_COL  +
    ", " + EVENT_TRACK_POS_COL;

  return SELECT_ALL_EVENTS_SQL;
}

const std::string &
query_manager
::select_all_event_ids_query() const
{
  static std::string SELECT_ALL_EVENT_IDS =
    " SELECT " + EVENT_ID_PK_COL  +
    " FROM "   + EVENT_TABLE_NAME +
    " WHERE "  + EVENT_SESSION_ID_COL +
    " =? "                           +
    " ORDER BY " + EVENT_ST_TIME_COL;

  return SELECT_ALL_EVENT_IDS;
}

const std::string &
query_manager
::select_events_by_type_query() const
{
  static std::string SELECT_EVENTS_BY_TYPE =
    "SELECT " + EVENT_ID_PK_COL            +
    " , "     + EVENT_EXT_ID_COL           +
    " , "     + EVENT_TYPE_COL +
    " , "     + EVENT_ST_TIME_COL +
    " , "     + EVENT_ST_FRAME_COL +
    " , "     + EVENT_END_TIME_COL +
    " , "     + EVENT_END_FRAME_COL +
    " , "     + EVENT_PROBABILITY_COL +
    " , "     + EVENT_BB_MIN_X_COL +
    " , "     + EVENT_BB_MIN_Y_COL +
    " , "     + EVENT_BB_MAX_X_COL +
    " , "     + EVENT_BB_MAX_Y_COL +
    " , "     + EVENT_DIRECTION_COL +
    " , "     + EVENT_DATE_CREATED_COL +
    " , "     + EVENT_DATE_MODIFIED_COL +

    " , "     + EVENT_TRACK_TRACK_ID_COL +
    " , "     + EVENT_TRACK_ST_TIME_COL +
    " , "     + EVENT_TRACK_ST_FRAME_COL +
    " , "     + EVENT_TRACK_END_TIME_COL +
    " , "     + EVENT_TRACK_END_FRAME_COL +
    " , "     + EVENT_TRACK_POS_COL +

    " FROM " + EVENT_TABLE_NAME +
    " INNER JOIN " + EVENT_TRACK_TABLE_NAME +
    " ON ( " + EVENT_ID_PK_COL  +
    " = " + EVENT_TRACK_EVENT_ID_COL   +
    " ) WHERE " + EVENT_TYPE_COL +
    " =? "  +
    " AND " + EVENT_SESSION_ID_COL +
    " =? "                           +
    " order by " + EVENT_ST_TIME_COL +
    " , "        + EVENT_TRACK_POS_COL;

  return SELECT_EVENTS_BY_TYPE;
}

const std::string &
query_manager
::delete_track_events_query() const
{
  static std::string DELETE_TRACK_EVENTS =
    "DELETE FROM " + EVENT_TRACK_TABLE_NAME +
    " WHERE " + EVENT_TRACK_EVENT_ID_COL +
    " =? ";

  return DELETE_TRACK_EVENTS;
}



const std::string &
query_manager
::insert_act_query() const
{
  //activity table access queries
  static std::string INSERT_ACT_SQL =
    "INSERT INTO " + ACT_TABLE_NAME   +
    " ( " + ACT_UUID_COL              +
    " , " + ACT_EXT_ID_COL            +
    " , " + ACT_SESSION_ID_COL        +
    " , " + ACT_TYPE_COL              +
    " , " + ACT_PROBABILITY_COL       +
    " , " + ACT_NORMALCY_COL          +
    " , " + ACT_SALIENCY_COL          +
    " , " + ACT_ST_TIME_COL           +
    " , " + ACT_ST_FRAME_COL          +
    " , " + ACT_END_TIME_COL          +
    " , " + ACT_END_FRAME_COL         +
    " , " + ACT_BB_MIN_X_COL  +
    " , " + ACT_BB_MIN_Y_COL  +
    " , " + ACT_BB_MAX_X_COL  +
    " , " + ACT_BB_MAX_Y_COL  +
    " ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)";

  return INSERT_ACT_SQL;
}

const std::string &
query_manager
::insert_act_event_query() const
{
  static std::string INSERT_ACT_EVENT_SQL =
    "INSERT INTO " + ACT_EVENT_TABLE_NAME  +
    " ( " + ACT_EVENT_ACT_ID_COL           +
    " , " + ACT_EVENT_EVENT_ID_COL         +
    " , " +    ACT_EVENT_POS_COL           +
    " ) VALUES (?,?,?)";

  return INSERT_ACT_EVENT_SQL;
}

const std::string &
query_manager
::select_act_by_id_query() const
{
  static std::string SELECT_ACT_BY_ID_SQL =
    "SELECT " + ACT_UUID_COL             +
    " , "     + ACT_EXT_ID_COL           +
    " , "     + ACT_TYPE_COL             +
    " , "     + ACT_PROBABILITY_COL      +
    " , "     + ACT_NORMALCY_COL         +
    " , "     + ACT_SALIENCY_COL +
    " , "     + ACT_ST_TIME_COL +
    " , "     + ACT_ST_FRAME_COL +
    " , "     + ACT_END_TIME_COL +
    " , "     + ACT_END_FRAME_COL +
    " , "     + ACT_BB_MIN_X_COL +
    " , "     + ACT_BB_MIN_Y_COL +
    " , "     + ACT_BB_MAX_X_COL +
    " , "     + ACT_BB_MAX_Y_COL +
    " , "     + ACT_DATE_CREATED_COL +
    " , "     + ACT_DATE_MODIFIED_COL +
    " , "     + ACT_EVENT_EVENT_ID_COL +
    " , "     + ACT_EVENT_POS_COL +

    " FROM " + ACT_TABLE_NAME +
    " INNER JOIN " + ACT_EVENT_TABLE_NAME +
    " ON ( " + ACT_ID_PK_COL  +
    " = " + ACT_EVENT_ACT_ID_COL   +
    " ) WHERE " + ACT_ID_PK_COL +
    " =? "  +
    " order by " + ACT_ID_PK_COL +
    " , " + ACT_EVENT_POS_COL;

  return SELECT_ACT_BY_ID_SQL;
}

const std::string &
query_manager
::select_act_by_uuid_query() const
{
  static std::string SELECT_ACT_BY_UUID_SQL =
    "SELECT " + ACT_UUID_COL             +
    " , "     + ACT_EXT_ID_COL           +
    " , "     + ACT_TYPE_COL             +
    " , "     + ACT_PROBABILITY_COL      +
    " , "     + ACT_NORMALCY_COL         +
    " , "     + ACT_SALIENCY_COL +
    " , "     + ACT_ST_TIME_COL +
    " , "     + ACT_ST_FRAME_COL +
    " , "     + ACT_END_TIME_COL +
    " , "     + ACT_END_FRAME_COL +
    " , "     + ACT_BB_MIN_X_COL +
    " , "     + ACT_BB_MIN_Y_COL +
    " , "     + ACT_BB_MAX_X_COL +
    " , "     + ACT_BB_MAX_Y_COL +
    " , "     + ACT_DATE_CREATED_COL +
    " , "     + ACT_DATE_MODIFIED_COL +
    " , "     + ACT_EVENT_EVENT_ID_COL +
    " , "     + ACT_EVENT_POS_COL +

    " FROM " + ACT_TABLE_NAME +
    " INNER JOIN " + ACT_EVENT_TABLE_NAME +
    " ON ( " + ACT_ID_PK_COL  +
    " = " + ACT_EVENT_ACT_ID_COL   +
    " ) WHERE " + ACT_UUID_COL +
    " =? "  +
    " order by " + ACT_ID_PK_COL +
    " , " + ACT_EVENT_POS_COL;

  return SELECT_ACT_BY_UUID_SQL;
}

const std::string &query_manager
::select_all_act_ids_query() const
{
  static std::string SELECT_ALL_ACT_IDS =
    " SELECT " + ACT_ID_PK_COL  +
    " FROM "   +  ACT_TABLE_NAME +
    " WHERE " + ACT_SESSION_ID_COL +
    " =? "
    " ORDER BY " + ACT_ST_TIME_COL;

  return SELECT_ALL_ACT_IDS;
}


const std::string &query_manager
::delete_act_events_query() const
{
  static std::string DELETE_ACT_EVENTS =
    "DELETE FROM " + ACT_EVENT_TABLE_NAME +
    " WHERE " + ACT_EVENT_ACT_ID_COL +
    " =? ";

  return DELETE_ACT_EVENTS;
}

const std::string &
query_manager
::event_id_by_uuid_query() const
{
  static std::string EVENT_ID_BY_UUID =
    "SELECT " + EVENT_ID_PK_COL  +
    " FROM  " + EVENT_TABLE_NAME +
    " WHERE " + EVENT_UUID_COL   +
    " =?";

  return EVENT_ID_BY_UUID;
}

const std::string &
query_manager
::get_track_column_select() const
{
  static std::string SELECT_TRACK_COLUMN_PORTION =
      TRACK_ID_PK_COL + " , "
    + TRACK_SESSION_ID_COL + " , "
    + TRACK_UUID_COL  + " , "
    + TRACK_EXT_ID_COL  + " , "
    + TRACK_LAST_MOD_MATCH_COL + " , "
    + TRACK_FALSE_ALARM_COL    + " , "
    + TRACK_STATUS_COL + " , "
    + TRACKER_ID_COL + " , "
    + TRACKER_TYPE_COL + " , "
    + TRACK_ATTRS_COL;

  return SELECT_TRACK_COLUMN_PORTION;
}

const std::string &
query_manager
::get_tile_md_column_select() const
{
  static std::string SELECT_TILE_MD_COLUMN_PORTION =
      VFM_FRAME_NUM_COL + " , "
    + VFM_FRAME_TIME_COL + " , "

    + VTM_UPPER_LEFT_LAT_COL + " , "
    + VTM_UPPER_LEFT_LON_COL  + " , "
    + VTM_UPPER_RIGHT_LAT_COL  + " , "
    + VTM_UPPER_RIGHT_LON_COL + " , "
    + VTM_LOWER_RIGHT_LAT_COL    + " , "
    + VTM_LOWER_RIGHT_LON_COL + " , "
    + VTM_LOWER_LEFT_LAT_COL + " , "
    + VTM_LOWER_LEFT_LON_COL + " , "
    + VTM_UPPER_LEFT_X_OFFSET_COL + " , "
    + VTM_UPPER_LEFT_Y_OFFSET_COL + " , "
    + VTM_PIXEL_WIDTH_COL + " , "
    + VTM_PIXEL_HEIGHT_COL;

  return SELECT_TILE_MD_COLUMN_PORTION;
}

const std::string &
query_manager
::get_track_state_column_select() const
{
  static std::string SELECT_STATE_COLUMN_PORTION =
    STATE_FRAME_NUM_COL +
    " , "     + STATE_FRAME_TIME_COL +
    " ,ST_X( "     + STATE_LOC_COL        + " ) as state_loc_x " +
    " ,ST_Y( "     + STATE_LOC_COL        + " ) as state_loc_y " +
    " ,ST_X( "     + STATE_WORLD_LOC_COL  + " ) as world_loc_x " +
    " ,ST_Y( "     + STATE_WORLD_LOC_COL  + " ) as world_loc_y " +
    " ,ST_X( "     + STATE_IMG_LOC_COL  + " ) as img_loc_x " +
    " ,ST_Y( "     + STATE_IMG_LOC_COL  + " ) as img_loc_y " +
    " ,ST_X( "     + STATE_LONLAT_COL     + " ) as state_lon " +
    " ,ST_Y( "     + STATE_LONLAT_COL     + " ) as state_lat " +

    " , "     + STATE_UTM_SMOOTHED_LOC_E_COL +
    " , "     + STATE_UTM_SMOOTHED_LOC_N_COL +
    " , "     + STATE_UTM_SMOOTHED_LOC_ZONE_COL +
    " , "     + STATE_UTM_SMOOTHED_LOC_IS_NORTH_COL +

    " , "     + STATE_UTM_RAW_LOC_E_COL +
    " , "     + STATE_UTM_RAW_LOC_N_COL +
    " , "     + STATE_UTM_RAW_LOC_ZONE_COL +
    " , "     + STATE_UTM_RAW_LOC_ZONE_IS_NORTH_COL +

    " , "     + STATE_UTM_VELOCITY_X_COL +
    " , "     + STATE_UTM_VELOCITY_Y_COL +


    " , "     + STATE_VELOCITY_X_COL +
    " , "     + STATE_VELOCITY_Y_COL +

    " , st_x(st_pointN(ST_Boundary( " + STATE_BBOX_COL + " ),1)) " +
    " , st_y(st_pointN(ST_Boundary( " + STATE_BBOX_COL + " ),1)) " +
    " , st_x(st_pointN(ST_Boundary( " + STATE_BBOX_COL + " ),3)) " +
    " , st_y(st_pointN(ST_Boundary( " + STATE_BBOX_COL + " ),3)) " +

    " , "     + STATE_ATTRS_COL +
    " , "     + STATE_AREA_COL +
    " , "     + STATE_IMAGE_CHIP_COL +
    " , "     + STATE_IMAGE_CHIP_OFFSET_COL  +
    " , "     + STATE_IMAGE_MASK_COL         +
    " , "     + STATE_MASK_i0_COL            +
    " , "     + STATE_MASK_j0_COL            +
    " , "     + STATE_COV_R0C0_COL           +
    " , "     + STATE_COV_R0C1_COL           +
    " , "     + STATE_COV_R1C0_COL           +
    " , "     + STATE_COV_R1C1_COL;


  return SELECT_STATE_COLUMN_PORTION;
}


const std::string &
query_manager
::get_pvo_column_select() const
{
  static std::string SELECT_PVO_COLUMN_PORTION =
    PVO_PERSON_PROBABILITY_COL +
    " , "     + PVO_VEHICLE_PROBABILITY_COL +
    " , "     + PVO_OTHER_PROBABILITY_COL;

  return SELECT_PVO_COLUMN_PORTION;
}

const std::string &
query_manager
::get_track_table_select() const
{
  static std::string SELECT_TRACK_TABLE_PORTION =
      TRACK_TABLE_NAME
    + " LEFT OUTER JOIN " + PVO_TABLE_NAME
    + " ON ( " + PVO_TRACK_ID_COL
    + " = " + TRACK_ID_PK_COL
    + " and pvo_frame_num = (select max ( pvo_frame_num) from pvo where  PVO_TRACK_ID = TRACK_ID)   ) "
    + " INNER JOIN " + STATE_TABLE_NAME
    + " ON ( " + STATE_TRACK_ID_COL
    + " = " + TRACK_ID_PK_COL
    + " )";

  return SELECT_TRACK_TABLE_PORTION;
}

std::string
query_manager
::make_in_clause( int size )
{
  std::string in_clause = " in (";

  for ( int i = 0; i < size - 1; i++)
  {
    in_clause += "?,";
  }

  in_clause += "?)";

  return in_clause;
}


template<typename T>
std::string
query_manager
::make_in_clause_by_value(const std::vector<T> & ids)
{
  std::string in_clause = " in (";
  unsigned id_size = ids.size();

  for ( unsigned i = 0; i < id_size - 1; i++)
  {
    in_clause += ( boost::lexical_cast< std::string >(ids[ i ]) + "," );
  }

  in_clause += (boost::lexical_cast< std::string >( ids[id_size - 1] ) + ")");

  return in_clause;
}

template std::string query_manager::make_in_clause_by_value(const std::vector<int> &);
template std::string query_manager::make_in_clause_by_value(const std::vector<long> &);

} // namespace vidtk

VBL_SMART_PTR_INSTANTIATE( vidtk::query_manager );
