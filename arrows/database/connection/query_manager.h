/*ckwg +5
 * Copyright 2012-2017 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */


#ifndef __QUERY_MANAGER_H__
#define __QUERY_MANAGER_H__

#include <vbl/vbl_smart_ptr.h>
#include <vbl/vbl_ref_count.h>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <string>
#include <iostream>
#include <vector>

namespace vidtk {

class query_manager;

typedef vbl_smart_ptr<query_manager> query_manager_sptr;

class query_manager
  : public vbl_ref_count
{

public:

  query_manager();
  virtual ~query_manager();
  virtual const std::string & insert_track_query() const = 0;
  virtual const std::string & insert_track_state_query() const = 0;
  virtual const std::string & insert_pvo_query() const;
  virtual const std::string & select_track_by_id_query() const;
  virtual const std::string & select_all_tracks_query() const;
  virtual const std::string & select_all_tracks_by_time_query() const;
  virtual const std::string & select_all_tracks_by_area_query() const;
  virtual const std::string & select_all_tracks_by_area_and_time_query() const;
  virtual const std::string & select_track_by_uuid_query() const;
  virtual const std::string & select_all_track_ids_query() const;
  virtual const std::string & select_max_track_ext_id_query() const;
  virtual const std::string & select_max_frame_number_query() const;
  virtual const std::string & new_select_trackids_by_last_frame_query() const;
  virtual const std::string & select_trackids_by_last_frame_query() const;
  virtual const std::string & select_tracks_by_frame_range_query() const;
  virtual const std::string select_tracks_by_frame_range_multi_session_query( int size ) ;
  virtual const std::string & select_active_tracks_by_frame_number_query() const;
  virtual const std::string select_active_tracks_by_frame_number_multi_session_query( int size );
  virtual const std::string & select_active_tracks_with_exclusion_session_query() const;
  virtual const std::string & insert_event_query() const;
  virtual const std::string & insert_event_track_query() const;
  virtual const std::string & track_uuid_by_id_query() const;
  virtual const std::string & track_id_by_uuid_query() const;
  virtual const std::string & select_event_by_id_query() const;
  virtual const std::string & select_event_by_uuid_query() const;
  virtual const std::string & select_all_events_query() const;
  virtual const std::string & select_terminated_events_query() const;
  virtual const std::string & select_all_event_ids_query() const;
  virtual const std::string & select_events_by_type_query() const;
  virtual const std::string & delete_track_events_query() const;

  virtual const std::string & insert_act_query() const;
  virtual const std::string & insert_act_event_query() const;
  virtual const std::string & select_act_by_id_query() const;
  virtual const std::string & select_act_by_uuid_query() const;
  virtual const std::string & select_all_act_ids_query() const;
  virtual const std::string & delete_act_events_query() const;
  virtual const std::string & event_id_by_uuid_query() const;

  /*
    The following 4 functions return the portion of the query that deals with each section.
    I prefer this approach to building the select queries so we don't have countless query
    changes every time the track/state/pvo structure changes.
   */
  virtual const std::string & get_track_column_select() const;
  virtual const std::string & get_track_state_column_select() const;
  virtual const std::string & get_pvo_column_select() const;
  virtual const std::string & get_tile_md_column_select() const;
  virtual const std::string & get_track_table_select() const;

  virtual std::string make_in_clause( int size );

  template<typename T>
  std::string make_in_clause_by_value( const std::vector<T> & ids);

  static const std::string POLYGON;

  //track_table definitions
  static const std::string VIDTK_DB_SCHEMA_NAME;
  static const std::string TRACK_TABLE_NAME;
  static const std::string TRACK_ID_PK_COL;
  static const std::string TRACK_UUID_COL;
  static const std::string TRACK_EXT_ID_COL;
  static const std::string TRACK_SESSION_ID_COL;
  static const std::string TRACK_LAST_MOD_MATCH_COL;
  static const std::string TRACK_FALSE_ALARM_COL;
  static const std::string TRACK_START_TIME_COL;
  static const std::string TRACK_END_TIME_COL ;
  static const std::string TRACK_REGION_COL;
  static const std::string TRACK_LOC_UPPER_LEFT_X_COL;
  static const std::string TRACK_LOC_UPPER_LEFT_Y_COL;
  static const std::string TRACK_LOC_LOWER_RIGHT_X_COL;
  static const std::string TRACK_LOC_LOWER_RIGHT_Y_COL;
  static const std::string TRACKER_ID_COL;
  static const std::string TRACKER_TYPE_COL;
  static const std::string TRACK_ATTRS_COL;
  static const std::string TRACK_STATUS_COL;
  static const std::string TRACK_DATE_CREATED_COL;
  static const std::string TRACK_DATE_MODIFIED_COL;
  static const std::string TRACK_ID_SEQ;

  //pvo table definitions
  static const std::string PVO_TABLE_NAME;
  static const std::string PVO_ID_PK_COL;
  static const std::string PVO_TRACK_ID_COL;
  static const std::string PVO_FRAME_NUM_COL;
  static const std::string PVO_PERSON_PROBABILITY_COL;
  static const std::string PVO_VEHICLE_PROBABILITY_COL;
  static const std::string PVO_OTHER_PROBABILITY_COL;

  //track_state table definitions
  static const std::string STATE_TABLE_NAME;
  static const std::string STATE_ID_PK_COL;
  static const std::string STATE_TRACK_ID_COL;
  static const std::string STATE_FRAME_NUM_COL;
  static const std::string STATE_FRAME_TIME_COL;
  static const std::string STATE_LOC_COL;
  static const std::string STATE_WORLD_LOC_COL;
  static const std::string STATE_IMG_LOC_COL;
  static const std::string STATE_LONLAT_COL;

  static const std::string STATE_UTM_SMOOTHED_LOC_E_COL;
  static const std::string STATE_UTM_SMOOTHED_LOC_N_COL;
  static const std::string STATE_UTM_SMOOTHED_LOC_ZONE_COL;
  static const std::string STATE_UTM_SMOOTHED_LOC_IS_NORTH_COL;
  static const std::string STATE_UTM_RAW_LOC_E_COL;
  static const std::string STATE_UTM_RAW_LOC_N_COL;
  static const std::string STATE_UTM_RAW_LOC_ZONE_COL;
  static const std::string STATE_UTM_RAW_LOC_ZONE_IS_NORTH_COL;
  static const std::string STATE_UTM_VELOCITY_X_COL;
  static const std::string STATE_UTM_VELOCITY_Y_COL;

  static const std::string STATE_VELOCITY_X_COL;
  static const std::string STATE_VELOCITY_Y_COL;
  static const std::string STATE_BBOX_COL;
  static const std::string STATE_AREA_COL;
  static const std::string STATE_ATTRS_COL;
  static const std::string STATE_IMAGE_CHIP_COL;
  static const std::string STATE_IMAGE_MASK_COL;
  static const std::string STATE_IMAGE_CHIP_OFFSET_COL;
  static const std::string STATE_MASK_i0_COL;
  static const std::string STATE_MASK_j0_COL;
  static const std::string STATE_SESSION_ID_COL;

  static const std::string STATE_COV_R0C0_COL;
  static const std::string STATE_COV_R0C1_COL;
  static const std::string STATE_COV_R1C0_COL;
  static const std::string STATE_COV_R1C1_COL ;

  //event table definitions
  static const std::string EVENT_TABLE_NAME;
  static const std::string EVENT_ID_PK_COL;
  static const std::string EVENT_UUID_COL;
  static const std::string EVENT_EXT_ID_COL;
  static const std::string EVENT_SESSION_ID_COL;
  static const std::string EVENT_TYPE_COL;
  static const std::string EVENT_ST_TIME_COL;
  static const std::string EVENT_END_TIME_COL;
  static const std::string EVENT_ST_FRAME_COL;
  static const std::string EVENT_END_FRAME_COL;
  static const std::string EVENT_PROBABILITY_COL;
  static const std::string EVENT_BB_MIN_X_COL;
  static const std::string EVENT_BB_MIN_Y_COL;
  static const std::string EVENT_BB_MAX_X_COL;
  static const std::string EVENT_BB_MAX_Y_COL;
  static const std::string EVENT_DIRECTION_COL;
  static const std::string EVENT_STATUS_COL;
  static const std::string EVENT_DATE_CREATED_COL;
  static const std::string EVENT_DATE_MODIFIED_COL;
  static const std::string EVENT_ID_SEQ;

  //event track table definitions
  static const std::string EVENT_TRACK_TABLE_NAME;
  static const std::string EVENT_TRACK_ID_PK_COL;
  static const std::string EVENT_TRACK_EVENT_ID_COL;
  static const std::string EVENT_TRACK_TRACK_ID_COL;
  static const std::string EVENT_TRACK_ST_TIME_COL;
  static const std::string EVENT_TRACK_END_TIME_COL;
  static const std::string EVENT_TRACK_ST_FRAME_COL;
  static const std::string EVENT_TRACK_END_FRAME_COL;
  static const std::string EVENT_TRACK_POS_COL;

  //activity table definitions
  static const std::string ACT_TABLE_NAME;
  static const std::string ACT_ID_PK_COL;
  static const std::string ACT_UUID_COL;
  static const std::string ACT_EXT_ID_COL;
  static const std::string ACT_SESSION_ID_COL;
  static const std::string ACT_TYPE_COL;
  static const std::string ACT_PROBABILITY_COL;
  static const std::string ACT_NORMALCY_COL;
  static const std::string ACT_SALIENCY_COL;
  static const std::string ACT_ST_TIME_COL;
  static const std::string ACT_ST_FRAME_COL;
  static const std::string ACT_END_TIME_COL;
  static const std::string ACT_END_FRAME_COL;
  static const std::string ACT_BB_MIN_X_COL;
  static const std::string ACT_BB_MIN_Y_COL;
  static const std::string ACT_BB_MAX_X_COL;
  static const std::string ACT_BB_MAX_Y_COL;
  static const std::string ACT_DATE_CREATED_COL;
  static const std::string ACT_DATE_MODIFIED_COL ;

  static const std::string ACT_ID_SEQ;

  //activity event table definitions
  static const std::string ACT_EVENT_TABLE_NAME;
  static const std::string ACT_EVENT_ACT_ID_COL;
  static const std::string ACT_EVENT_EVENT_ID_COL;
  static const std::string ACT_EVENT_POS_COL;

  //frame_metadata definitions
  static const std::string VIDTK_FRAME_METADATA_TABLE_NAME;
  static const std::string VFM_TS_ID_COL;
  static const std::string VFM_MISSION_ID_COL;
  static const std::string VFM_FRAME_NUM_COL;
  static const std::string VFM_FRAME_TIME_COL;
  static const std::string VFM_FILE_PATH_COL;

  static const std::string VIDTK_FRAME_PRODUCTS_TABLE_NAME;
  static const std::string VFP_ID_COL;
  static const std::string VFP_SESSION_ID_COL;
  static const std::string VFP_VIDTK_TS_COL;
  static const std::string VFP_SRC_TO_REF_REF_TS_COL;
  static const std::string VFP_SRC_TO_REF_SRC_TS_COL;
  static const std::string VFP_SRC_TO_REF_H_R0C0_COL;
  static const std::string VFP_SRC_TO_REF_H_R0C1_COL;
  static const std::string VFP_SRC_TO_REF_H_R0C2_COL;
  static const std::string VFP_SRC_TO_REF_H_R1C0_COL;
  static const std::string VFP_SRC_TO_REF_H_R1C1_COL;
  static const std::string VFP_SRC_TO_REF_H_R1C2_COL;
  static const std::string VFP_SRC_TO_REF_H_R2C0_COL;
  static const std::string VFP_SRC_TO_REF_H_R2C1_COL;
  static const std::string VFP_SRC_TO_REF_H_R2C2_COL;
  static const std::string VFP_SRC_TO_REF_IS_VALID_COL;
  static const std::string VFP_SRC_TO_REF_IS_NEW_REF_COL;
  static const std::string VFP_SRC_TO_UTM_ZONE_COL;
  static const std::string VFP_SRC_TO_UTM_NORTHING_COL;
  static const std::string VFP_SRC_TO_UTM_SRC_TS_COL;
  static const std::string VFP_SRC_TO_UTM_H_R0C0_COL;
  static const std::string VFP_SRC_TO_UTM_H_R0C1_COL;
  static const std::string VFP_SRC_TO_UTM_H_R0C2_COL;
  static const std::string VFP_SRC_TO_UTM_H_R1C0_COL;
  static const std::string VFP_SRC_TO_UTM_H_R1C1_COL;
  static const std::string VFP_SRC_TO_UTM_H_R1C2_COL;
  static const std::string VFP_SRC_TO_UTM_H_R2C0_COL;
  static const std::string VFP_SRC_TO_UTM_H_R2C1_COL;
  static const std::string VFP_SRC_TO_UTM_H_R2C2_COL;
  static const std::string VFP_SRC_TO_UTM_IS_VALID_COL;
  static const std::string VFP_SRC_TO_UTM_IS_NEW_COL;
  static const std::string VFP_COMPUTED_GSD_COL;

  static const std::string VIDTK_TILE_METADATA_TABLE_NAME;
  static const std::string VTM_ID_COL;
  static const std::string VTM_SESSION_ID_COL;
  static const std::string VTM_VIDTK_TS_COL;
  static const std::string VTM_UPPER_LEFT_LAT_COL;
  static const std::string VTM_UPPER_LEFT_LON_COL;
  static const std::string VTM_UPPER_RIGHT_LAT_COL;
  static const std::string VTM_UPPER_RIGHT_LON_COL;
  static const std::string VTM_LOWER_RIGHT_LAT_COL;
  static const std::string VTM_LOWER_RIGHT_LON_COL;
  static const std::string VTM_LOWER_LEFT_LAT_COL;
  static const std::string VTM_LOWER_LEFT_LON_COL;
  static const std::string VTM_UPPER_LEFT_X_OFFSET_COL;
  static const std::string VTM_UPPER_LEFT_Y_OFFSET_COL;
  static const std::string VTM_PIXEL_WIDTH_COL;
  static const std::string VTM_PIXEL_HEIGHT_COL;

protected:

private:

}; //class query_manager


//set the size of the maximum allowable resultset
#define MAX_QUERY_SIZE 10000

typedef boost::uuids::uuid uuid_t;

enum db_type { SQLITE, MYSQL, POSTGRES };


//pod table definitions
static const std::string STATE_POD_TABLE_NAME = "STATE_POD";
static const std::string STATE_POD_ID_PK_COL = "STATE_POD_ID";
static const std::string STATE_POD_VERSION_COL = "STATE_POD_VERSION";
static const std::string STATE_POD_PARENT_UUID_COL = "STATE_POD_PARENT_UUID";
static const std::string STATE_POD_NAME_COL = "STATE_POD_NAME";
static const std::string STATE_POD_VENDOR_COL = "STATE_POD_VENDOR";
static const std::string STATE_POD_TYPE_COL = "STATE_POD_TYPE";
static const std::string STATE_POD_DESCRIPTION_COL = "STATE_POD_DESC";
static const std::string STATE_POD_CHUNK_SIZE_COL = "STATE_POD_CHUNK_SIZE";
static const std::string STATE_POD_CHUNK_COL = "STATE_POD_CHUNK";

//pod table creation
static const std::string CREATE_STATE_POD_TABLE =
  "CREATE TABLE " + STATE_POD_TABLE_NAME +
  " ( "
  + STATE_POD_ID_PK_COL       + " INTEGER PRIMARY KEY, "
  + STATE_POD_VERSION_COL     + " TEXT, "
  + STATE_POD_NAME_COL        + " TEXT, "
  + STATE_POD_VENDOR_COL      + " TEXT, "
  + STATE_POD_TYPE_COL        + " INTEGER, "
  + STATE_POD_DESCRIPTION_COL + " TEXT, "
  + STATE_POD_CHUNK_SIZE_COL  + " INTEGER, "
  + STATE_POD_CHUNK_COL       + " BLOB );";


//pod table access queries
static const std::string INSERT_STATE_POD_SQL =
  "INSERT INTO " + STATE_POD_TABLE_NAME  +
  " ( " + STATE_POD_VERSION_COL          +
  " , " + STATE_POD_NAME_COL             +
  " , " + STATE_POD_VENDOR_COL           +
  " , " + STATE_POD_TYPE_COL             +
  " , " + STATE_POD_DESCRIPTION_COL      +
  " , " + STATE_POD_CHUNK_SIZE_COL       +
  " , " + STATE_POD_CHUNK_COL            +
  " ) VALUES ( ?,?,?,?,?,?,?)";

//pod table access queries
static const std::string UPDATE_STATE_POD_SQL =
  "UPDATE " + STATE_POD_TABLE_NAME    +
  " SET " + STATE_POD_VERSION_COL     + " =? " +
  " ,   " + STATE_POD_NAME_COL        + " =? " +
  " ,   " + STATE_POD_VENDOR_COL      + " =? " +
  " ,   " + STATE_POD_TYPE_COL        + " =? " +
  " ,   " + STATE_POD_DESCRIPTION_COL + " =? " +
  " ,   " + STATE_POD_CHUNK_SIZE_COL  + " =? " +
  " ,   " + STATE_POD_CHUNK_COL       + " =? " +

  " WHERE " + STATE_POD_NAME_COL    + " =? " +
  " AND   " + STATE_POD_VENDOR_COL  + " =? " ;

static const std::string SELECT_STATE_POD_SQL =
  "SELECT " + STATE_POD_VERSION_COL  +
  " , " + STATE_POD_NAME_COL +
  " , " + STATE_POD_VENDOR_COL +
  " , " + STATE_POD_TYPE_COL +
  " , " + STATE_POD_DESCRIPTION_COL +
  " , " + STATE_POD_CHUNK_SIZE_COL +
  " , " + STATE_POD_CHUNK_COL +

  " FROM "   + STATE_POD_TABLE_NAME +
  " WHERE "  + STATE_POD_NAME_COL   +
  " =? AND " + STATE_POD_VENDOR_COL +
  " =? ";


//mission table definitions
static const std::string MISSION_TABLE_NAME = "MISSION";
static const std::string MISSION_ID_COL = "MISSION_ID";
static const std::string MISSION_NAME_COL = "MISSION_NAME";
static const std::string MISSION_KEY_WORDS_COL = "MISSION_KEY_WORDS";
static const std::string MISSION_IMAGE_PATH_COL = "IMAGE_PATH";
static const std::string MISSION_IMAGE_FILE_PATTERN_COL = "IMAGE_FILE_PATTERN";
static const std::string MRJ_FILE_PATH_COL = "MRJ_FILE_PATH";
static const std::string TS_CACHE_FILE_PATH_COL = "TS_CACHE_FILE_PATH";

static const std::string MISSION_START_TIME_COL = "MISSION_START_TIME";
static const std::string MISSION_END_TIME_COL = "MISSION_END_TIME";
static const std::string MISSION_START_FRAME_COL = "MISSION_START_FRAME";
static const std::string MISSION_END_FRAME_COL = "MISSION_END_FRAME";

static const std::string MISSION_BBOX_UPPER_LEFT_LAT_COL = "MISSION_UPPER_LEFT_LAT";
static const std::string MISSION_BBOX_UPPER_LEFT_LON_COL = "MISSION_UPPER_LEFT_LON";
static const std::string MISSION_BBOX_UPPER_RIGHT_LAT_COL = "MISSION_UPPER_RIGHT_LAT";
static const std::string MISSION_BBOX_UPPER_RIGHT_LON_COL = "MISSION_UPPER_RIGHT_LON";
static const std::string MISSION_BBOX_LOWER_LEFT_LAT_COL = "MISSION_LOWER_LEFT_LAT";
static const std::string MISSION_BBOX_LOWER_LEFT_LON_COL = "MISSION_LOWER_LEFT_LON";
static const std::string MISSION_BBOX_LOWER_RIGHT_LAT_COL = "MISSION_LOWER_RIGHT_LAT";
static const std::string MISSION_BBOX_LOWER_RIGHT_LON_COL = "MISSION_LOWER_RIGHT_LON";
static const std::string MISSION_MODEL_TIEPOINT_COL = "MODEL_TIE_POINT";
static const std::string MISSION_UTM_ZONE_COL = "UTM_ZONE";
static const std::string IMAGE_HEIGHT_COL = "IMAGE_HEIGHT";
static const std::string IMAGE_WIDTH_COL = "IMAGE_WIDTH";
static const std::string MISSION_SENSOR_COL = "MISSION_SENSOR";
static const std::string PIXEL_SCALE_COL = "PIXEL_SCALE";
static const std::string MISSION_GSD_COL = "MISSION_GSD";
static const std::string MISSION_GSD_X_COL = "MISSION_GSD_X";
static const std::string MISSION_GSD_Y_COL = "MISSION_GSD_Y";

static const std::string MISSION_FRAME_TIME_UNITS_COL = "MISSION_FRAME_TIME_UNITS";
static const std::string MISSION_FRAME_COUNT_COL = "FRAME_COUNT";
static const std::string MISSION_TIMEZONE_COL = "MISSION_TIMEZONE";
static const std::string MISSION_THUMBNAIL_COL = "MISSION_THUMBNAIL";
static const std::string MISSION_DESCRIPTION_COL = "MISSION_DESCRIPTION";
static const std::string MISSION_DATE_CREATED_COL = "MISSION_DATE_CREATED";
static const std::string MISSION_ID_SEQ = "MISSION_MISSION_ID_SEQ";

//mission table creation
static const std::string CREATE_MISSION_TABLE =
  "CREATE TABLE " + MISSION_TABLE_NAME +
  " ( "
  + MISSION_ID_COL                    + " INTEGER PRIMARY KEY UNIQUE, "
  + MISSION_NAME_COL                  + " TEXT, "
  + MISSION_DESCRIPTION_COL           + " TEXT, "
  + MISSION_KEY_WORDS_COL             + " TEXT, "
  + MISSION_IMAGE_PATH_COL            + " TEXT, "
  + MISSION_IMAGE_FILE_PATTERN_COL    + " TEXT, "
  + MRJ_FILE_PATH_COL                 + " TEXT, "
  + TS_CACHE_FILE_PATH_COL            + " TEXT, "
  + MISSION_START_TIME_COL            + " REAL, "
  + MISSION_END_TIME_COL              + " REAL, "
  + MISSION_START_FRAME_COL           + " INTEGER, "
  + MISSION_END_FRAME_COL             + " INTEGER, "
  + MISSION_BBOX_UPPER_LEFT_LAT_COL   + " REAL, "
  + MISSION_BBOX_UPPER_LEFT_LON_COL   + " REAL, "
  + MISSION_BBOX_UPPER_RIGHT_LAT_COL  + " REAL, "
  + MISSION_BBOX_UPPER_RIGHT_LON_COL  + " REAL, "
  + MISSION_BBOX_LOWER_LEFT_LAT_COL   + " REAL, "
  + MISSION_BBOX_LOWER_LEFT_LON_COL   + " REAL, "
  + MISSION_BBOX_LOWER_RIGHT_LAT_COL  + " REAL, "
  + MISSION_BBOX_LOWER_RIGHT_LON_COL  + " REAL, "
  + MISSION_MODEL_TIEPOINT_COL        + " TEXT, "
  + MISSION_UTM_ZONE_COL              + " TEXT, "
  + IMAGE_HEIGHT_COL                  + " INTEGER, "
  + IMAGE_WIDTH_COL                   + " INTEGER, "
  + PIXEL_SCALE_COL                   + " TEXT, "
  + MISSION_SENSOR_COL                + " TEXT, "
  + MISSION_GSD_X_COL                 + " REAL, "
  + MISSION_GSD_Y_COL                 + " REAL, "
  + MISSION_FRAME_TIME_UNITS_COL      + " TEXT, "
  + MISSION_FRAME_COUNT_COL           + " TEXT, "
  + MISSION_TIMEZONE_COL              + " TEXT, "
  + MISSION_THUMBNAIL_COL             + " BLOB, "
  + MISSION_DATE_CREATED_COL          + " DATE); ";


static const std::string INSERT_MISSION_SQL =
  " INSERT INTO " + MISSION_TABLE_NAME     +
  " ( " + MISSION_NAME_COL                 +
  " , " + MISSION_KEY_WORDS_COL            +
  " , " + MISSION_IMAGE_PATH_COL           +
  " , " + MISSION_IMAGE_FILE_PATTERN_COL   +
  " , " + MRJ_FILE_PATH_COL                +
  " , " + TS_CACHE_FILE_PATH_COL           +
  " , " + MISSION_START_TIME_COL           +
  " , " + MISSION_END_TIME_COL             +
  " , " + MISSION_START_FRAME_COL          +
  " , " + MISSION_END_FRAME_COL            +
  " , " + MISSION_BBOX_UPPER_LEFT_LAT_COL  +
  " , " + MISSION_BBOX_UPPER_LEFT_LON_COL  +
  " , " + MISSION_BBOX_UPPER_RIGHT_LAT_COL +
  " , " + MISSION_BBOX_UPPER_RIGHT_LON_COL +
  " , " + MISSION_BBOX_LOWER_LEFT_LAT_COL  +
  " , " + MISSION_BBOX_LOWER_LEFT_LON_COL  +
  " , " + MISSION_BBOX_LOWER_RIGHT_LAT_COL +
  " , " + MISSION_BBOX_LOWER_RIGHT_LON_COL +
  " , " + MISSION_MODEL_TIEPOINT_COL       +
  " , " + MISSION_UTM_ZONE_COL             +
  " , " + IMAGE_HEIGHT_COL                 +
  " , " + IMAGE_WIDTH_COL                  +
  " , " + PIXEL_SCALE_COL                  +
  " , " + MISSION_SENSOR_COL               +
  " , " + MISSION_GSD_X_COL                +
  " , " + MISSION_GSD_Y_COL                +
  " , " + MISSION_FRAME_TIME_UNITS_COL     +
  " , " + MISSION_FRAME_COUNT_COL          +
  " , " + MISSION_TIMEZONE_COL             +
  " , " + MISSION_DESCRIPTION_COL          +
  " , " + MISSION_DATE_CREATED_COL         +
  " ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)";


static const std::string UPDATE_MISSION_SQL =
  "UPDATE " + MISSION_TABLE_NAME           +
  " SET "   + MISSION_NAME_COL             + " =? " +
  " , " + MISSION_KEY_WORDS_COL            + " =? " +
  " , " + MISSION_IMAGE_PATH_COL           + " =? " +
  " , " + MISSION_IMAGE_FILE_PATTERN_COL   + " =? " +
  " , " + MRJ_FILE_PATH_COL                + " =? " +
  " , " + TS_CACHE_FILE_PATH_COL           + " =? " +
  " , " + MISSION_START_TIME_COL           + " =? " +
  " , " + MISSION_END_TIME_COL             + " =? " +
  " , " + MISSION_START_FRAME_COL          + " =? " +
  " , " + MISSION_END_FRAME_COL            + " =? " +
  " , " + MISSION_BBOX_UPPER_LEFT_LAT_COL  + " =? " +
  " , " + MISSION_BBOX_UPPER_LEFT_LON_COL  + " =? " +
  " , " + MISSION_BBOX_UPPER_RIGHT_LAT_COL + " =? " +
  " , " + MISSION_BBOX_UPPER_RIGHT_LON_COL + " =? " +
  " , " + MISSION_BBOX_LOWER_LEFT_LAT_COL  + " =? " +
  " , " + MISSION_BBOX_LOWER_LEFT_LON_COL  + " =? " +
  " , " + MISSION_BBOX_LOWER_RIGHT_LAT_COL + " =? " +
  " , " + MISSION_BBOX_LOWER_RIGHT_LON_COL + " =? " +
  " , " + MISSION_MODEL_TIEPOINT_COL       + " =? " +
  " , " + MISSION_UTM_ZONE_COL             + " =? " +
  " , " + IMAGE_HEIGHT_COL                 + " =? " +
  " , " + IMAGE_WIDTH_COL                  + " =? " +
  " , " + PIXEL_SCALE_COL                  + " =? " +
  " , " + MISSION_SENSOR_COL               + " =? " +
  " , " + MISSION_GSD_X_COL                + " =? " +
  " , " + MISSION_GSD_Y_COL                + " =? " +
  " , " + MISSION_FRAME_TIME_UNITS_COL     + " =? " +
  " , " + MISSION_FRAME_COUNT_COL          + " =? " +
  " , " + MISSION_TIMEZONE_COL             + " =? " +
  " , " + MISSION_DESCRIPTION_COL          + " =? " +
  " WHERE " + MISSION_ID_COL               + " =?";

static const std::string SELECT_MISSION_ALL_SQL =
  "SELECT " + MISSION_ID_COL                   +
  " , "     + MISSION_NAME_COL                 +
  " , "     + MISSION_KEY_WORDS_COL            +
  " , "     + MISSION_IMAGE_PATH_COL           +
  " , "     + MISSION_IMAGE_FILE_PATTERN_COL   +
  " , "     + MRJ_FILE_PATH_COL                +
  " , "     + TS_CACHE_FILE_PATH_COL           +
  " , "     + MISSION_START_TIME_COL           +
  " , "     + MISSION_END_TIME_COL             +
  " , "     + MISSION_START_FRAME_COL          +
  " , "     + MISSION_END_FRAME_COL            +
  " , "     + MISSION_BBOX_UPPER_LEFT_LAT_COL  +
  " , "     + MISSION_BBOX_UPPER_LEFT_LON_COL  +
  " , "     + MISSION_BBOX_UPPER_RIGHT_LAT_COL +
  " , "     + MISSION_BBOX_UPPER_RIGHT_LON_COL +
  " , "     + MISSION_BBOX_LOWER_LEFT_LAT_COL  +
  " , "     + MISSION_BBOX_LOWER_LEFT_LON_COL  +
  " , "     + MISSION_BBOX_LOWER_RIGHT_LAT_COL +
  " , "     + MISSION_BBOX_LOWER_RIGHT_LON_COL +
  " , "     + MISSION_MODEL_TIEPOINT_COL       +
  " , "     + MISSION_UTM_ZONE_COL             +
  " , "     + IMAGE_HEIGHT_COL                 +
  " , "     + IMAGE_WIDTH_COL                  +
  " , "     + PIXEL_SCALE_COL                  +
  " , "     + MISSION_SENSOR_COL               +
  " , "     + MISSION_GSD_X_COL                +
  " , "     + MISSION_GSD_Y_COL                +
  " , "     + MISSION_FRAME_TIME_UNITS_COL     +
  " , "     + MISSION_FRAME_COUNT_COL          +
  " , "     + MISSION_TIMEZONE_COL             +
  " , "     + MISSION_THUMBNAIL_COL            +
  " , "     + MISSION_DESCRIPTION_COL          +
  " FROM "  + MISSION_TABLE_NAME;

static const std::string SELECT_MISSION_SQL =
  SELECT_MISSION_ALL_SQL + " WHERE " + MISSION_ID_COL + " = ?";

static const std::string SELECT_MISSION_BY_NAME_SQL =
  SELECT_MISSION_ALL_SQL + " WHERE " + MISSION_NAME_COL + " = ?";


//aoi table definitions
static const std::string AOI_TABLE_NAME = "AOI";
static const std::string AOI_ID_COL = "AOI_ID";
static const std::string AOI_MISSION_ID_COL = "AOI_MISSION_ID";
static const std::string AOI_NAME_COL = "AOI_NAME";
static const std::string CROP_STRING_COL = "CROP_STRING";
static const std::string AOI_UPPER_LEFT_LAT_COL = "AOI_UPPER_LEFT_LAT";
static const std::string AOI_UPPER_LEFT_LON_COL = "AOI_UPPER_LEFT_LON";
static const std::string AOI_UPPER_RIGHT_LAT_COL = "AOI_UPPER_RIGHT_LAT";
static const std::string AOI_UPPER_RIGHT_LON_COL = "AOI_UPPER_RIGHT_LON";
static const std::string AOI_LOWER_LEFT_LAT_COL = "AOI_LOWER_LEFT_LAT";
static const std::string AOI_LOWER_LEFT_LON_COL = "AOI_LOWER_LEFT_LON";
static const std::string AOI_LOWER_RIGHT_LAT_COL = "AOI_LOWER_RIGHT_LAT";
static const std::string AOI_LOWER_RIGHT_LON_COL = "AOI_LOWER_RIGHT_LON";
static const std::string AOI_START_TIME_COL = "AOI_START_TIME";
static const std::string AOI_END_TIME_COL = "AOI_END_TIME";
static const std::string AOI_START_FRAME_COL = "AOI_START_FRAME";
static const std::string AOI_END_FRAME_COL = "AOI_END_FRAME";
static const std::string AOI_DESCRIPTION_COL = "AOI_DESCRIPTION";
static const std::string AOI_ID_SEQ = "AOI_AOI_ID_SEQ";


//aoi creation (sqlite) and queries
static const std::string CREATE_AOI_TABLE =
  "CREATE TABLE " + AOI_TABLE_NAME +
  " ( "
  + AOI_ID_COL         + " INTEGER PRIMARY KEY UNIQUE, "
  + AOI_MISSION_ID_COL + " INTEGER, "
  + AOI_NAME_COL + " TEXT, "
  + CROP_STRING_COL + " TEXT, "
  + AOI_UPPER_LEFT_LAT_COL + " REAL, "
  + AOI_UPPER_LEFT_LON_COL + " REAL, "
  + AOI_UPPER_RIGHT_LAT_COL + " REAL, "
  + AOI_UPPER_RIGHT_LON_COL + " REAL, "
  + AOI_LOWER_LEFT_LAT_COL + " REAL, "
  + AOI_LOWER_LEFT_LON_COL + " REAL, "
  + AOI_LOWER_RIGHT_LAT_COL + " REAL, "
  + AOI_LOWER_RIGHT_LON_COL + " REAL, "
  + AOI_START_TIME_COL + " REAL, "
  + AOI_END_TIME_COL + " REAL,  "
  + AOI_START_FRAME_COL + " INTEGER, "
  + AOI_END_FRAME_COL + " INTEGER, "
  + AOI_DESCRIPTION_COL + " TEXT);";


static const std::string SELECT_AOI_SQL =
  "SELECT " + AOI_ID_COL +
  " , "     + AOI_MISSION_ID_COL +
  " , "     + AOI_NAME_COL +
  " , "     + CROP_STRING_COL +
  " , "     + AOI_UPPER_LEFT_LAT_COL +
  " , "     + AOI_UPPER_LEFT_LON_COL +
  " , "     + AOI_UPPER_RIGHT_LAT_COL  +
  " , "     + AOI_UPPER_RIGHT_LON_COL  +
  " , "     + AOI_LOWER_LEFT_LAT_COL +
  " , "     + AOI_LOWER_LEFT_LON_COL +
  " , "     + AOI_LOWER_RIGHT_LAT_COL +
  " , "     + AOI_LOWER_RIGHT_LON_COL +
  " , "     + AOI_START_TIME_COL +
  " , "     + AOI_END_TIME_COL +
  " , "     + AOI_START_FRAME_COL +
  " , "     + AOI_END_FRAME_COL +
  " , "     + AOI_DESCRIPTION_COL +
  " FROM "  + AOI_TABLE_NAME +
  " WHERE " + AOI_ID_COL +
  " =?";

static const std::string SELECT_AOI_BY_MISSION_SQL =
  "SELECT " + AOI_ID_COL +
  " , "     + AOI_MISSION_ID_COL +
  " , "     + AOI_NAME_COL +
  " , "     + CROP_STRING_COL +
  " , "     + AOI_UPPER_LEFT_LAT_COL +
  " , "     + AOI_UPPER_LEFT_LON_COL +
  " , "     + AOI_UPPER_RIGHT_LAT_COL  +
  " , "     + AOI_UPPER_RIGHT_LON_COL  +
  " , "     + AOI_LOWER_LEFT_LAT_COL +
  " , "     + AOI_LOWER_LEFT_LON_COL +
  " , "     + AOI_LOWER_RIGHT_LAT_COL +
  " , "     + AOI_LOWER_RIGHT_LON_COL +
  " , "     + AOI_START_TIME_COL +
  " , "     + AOI_END_TIME_COL +
  " , "     + AOI_START_FRAME_COL +
  " , "     + AOI_END_FRAME_COL +
  " , "     + AOI_DESCRIPTION_COL +
  " FROM "  + AOI_TABLE_NAME +
  " WHERE " + AOI_MISSION_ID_COL +
  " =?";

static const std::string INSERT_AOI_SQL =
  "INSERT INTO " + AOI_TABLE_NAME +
    " ( "
  + AOI_MISSION_ID_COL + " , "
  + AOI_NAME_COL + " , "
  + CROP_STRING_COL + " , "
  + AOI_UPPER_LEFT_LAT_COL + " , "
  + AOI_UPPER_LEFT_LON_COL + " , "
  + AOI_UPPER_RIGHT_LAT_COL  + " , "
  + AOI_UPPER_RIGHT_LON_COL  + " , "
  + AOI_LOWER_LEFT_LAT_COL + " , "
  + AOI_LOWER_LEFT_LON_COL + " , "
  + AOI_LOWER_RIGHT_LAT_COL + " , "
  + AOI_LOWER_RIGHT_LON_COL + " , "
  + AOI_START_TIME_COL + " , "
  + AOI_END_TIME_COL + " , "
  + AOI_START_FRAME_COL + " , "
  + AOI_END_FRAME_COL + " , "
  + AOI_DESCRIPTION_COL
  + " ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)";


//session table definitions
static const std::string SESSION_METADATA_TABLE_NAME = "SESSION_METADATA";
static const std::string SESSION_METADATA_SEQUENCE = "session_metadata_session_meta_id_seq";

static const std::string SESSION_META_ID_PK_COL = "SESSION_META_ID";
static const std::string SESSION_AOI_ID_COL = "SESSION_AOI_ID";
static const std::string PARENT_SESSION_ID_COL = "PARENT_SESSION_ID";
static const std::string SESSION_TYPE_COL = "SESSION_TYPE";
static const std::string IS_GROUND_TRUTH_COL = "IS_GROUNDTRUTH";
static const std::string PERSEAS_SHA1_COL = "PERSEAS_SHA1";
static const std::string PERSEAS_STATUS_COL = "PERSEAS_STATUS";
static const std::string APPS_SHA1_COL= "APPS_SHA1";
static const std::string APPS_STATUS_COL = "APPS_STATUS";
static const std::string DATA_SHA1_COL = "DATA_SHA1";
static const std::string DATA_STATUS_COL = "DATA_STATUS";
static const std::string QTTESTING_SHA1_COL = "QTTESTING_SHA1";
static const std::string QTTESTING_STATUS_COL = "QTTESTING_STATUS";
static const std::string VIDTK_SHA1_COL = "VIDTK_SHA1";
static const std::string VIDTK_STATUS_COL = "VIDTK_STATUS";
static const std::string VISGUI_SHA1_COL = "VISGUI_SHA1";
static const std::string VISGUI_STATUS_COL = "VISGUI_STATUS";
static const std::string VXL_SHA1_COL = "VXL_SHA1";
static const std::string VXL_STATUS_COL = "VXL_STATUS";
static const std::string SECURITY_CLASSIFICATION_COL = "SECURITY_CLASSIFICATION";
static const std::string SECURITY_POLICY_NAME_COL = "SECURITY_POLICYNAME";
static const std::string SECURITY_CONTROL_SYSTEM_COL = "SECURITY_CONTROLSYSTEM";
static const std::string SECURITY_DISSEMINATION_COL = "SECURITY_DISSEMINATION";
static const std::string SECURITY_RELEASABILITY_COL = "SECURITY_RELEASABILITY";
static const std::string SESSION_DESCRIPTION_COL = "SESSION_DESCRIPTION";
static const std::string SESSION_MODIFIED_BY_COL = "SESSION_MODIFIED_BY";
static const std::string DATE_CREATED_COL = "DATE_CREATED";

//session_params table definitions
static const std::string SESSION_PARAMS_TABLE_NAME = "SESSION_PARAMS";
static const std::string SESSION_PARAMS_ID_PK_COL = "SESSION_PARAMS_ID";
static const std::string PARAMS_SESSION_META_ID_COL = "PARAMS_SESSION_META_ID";
static const std::string PARAM_NAME_COL = "PARAM_NAME";
static const std::string PARAM_VALUE_COL = "PARAM_VALUE";

//session creation (sqlite) and queries
static const std::string CREATE_SESSION_METADATA_TABLE =
  "CREATE TABLE " + SESSION_METADATA_TABLE_NAME +
  " ( "
  + SESSION_META_ID_PK_COL       + " INTEGER PRIMARY KEY UNIQUE, "
  + SESSION_AOI_ID_COL           + " INTEGER, "
  + PARENT_SESSION_ID_COL        + " INTEGER, "
  + SESSION_TYPE_COL         + " TEXT, "
  + IS_GROUND_TRUTH_COL      + " BOOL, "
  + PERSEAS_SHA1_COL        + " TEXT, "
  + PERSEAS_STATUS_COL      + " TEXT, "
  + APPS_SHA1_COL           + " TEXT, "
  + APPS_STATUS_COL         + " TEXT, "
  + DATA_SHA1_COL           + " TEXT, "
  + DATA_STATUS_COL         + " TEXT, "
  + QTTESTING_SHA1_COL     + " TEXT, "
  + QTTESTING_STATUS_COL   + " TEXT, "
  + VIDTK_SHA1_COL         + " TEXT, "
  + VIDTK_STATUS_COL       + " TEXT, "
  + VISGUI_SHA1_COL        + " TEXT, "
  + VISGUI_STATUS_COL      + " TEXT, "
  + VXL_SHA1_COL           + " TEXT, "
  + VXL_STATUS_COL         + " TEXT, "
  + SECURITY_CLASSIFICATION_COL + " TEXT, "
  + SECURITY_POLICY_NAME_COL + " TEXT, "
  + SECURITY_CONTROL_SYSTEM_COL + " TEXT, "
  + SECURITY_DISSEMINATION_COL + " TEXT, "
  + SECURITY_RELEASABILITY_COL + " TEXT, "
  + SESSION_DESCRIPTION_COL + " TEXT, "
  + SESSION_MODIFIED_BY_COL + " TEXT, "
  + DATE_CREATED_COL       + " DATE ); ";



static const std::string CREATE_SESSION_PARAMS_TABLE =
 "CREATE TABLE " + SESSION_PARAMS_TABLE_NAME +
  " ( "
  + SESSION_PARAMS_ID_PK_COL    + " INTEGER PRIMARY KEY UNIQUE, "
  + PARAMS_SESSION_META_ID_COL  + " INTEGER, "
  + PARAM_NAME_COL              + " TEXT, "
  + PARAM_VALUE_COL             + " TEXT);";

static const std::string INSERT_TRACK_SESSION_METADATA =
  "INSERT INTO " + SESSION_METADATA_TABLE_NAME +
  " ( " + SESSION_AOI_ID_COL        +
  " , " + SESSION_TYPE_COL         +
  " , " + IS_GROUND_TRUTH_COL         +
  " , " + PERSEAS_SHA1_COL +
  " , " + PERSEAS_STATUS_COL +
  " , " + APPS_SHA1_COL +
  " , " + APPS_STATUS_COL +
  " , " + DATA_SHA1_COL +
  " , " + DATA_STATUS_COL +
  " , " + QTTESTING_SHA1_COL +
  " , " + QTTESTING_STATUS_COL +
  " , " + VIDTK_SHA1_COL +
  " , " + VIDTK_STATUS_COL +
  " , " + VISGUI_SHA1_COL +
  " , " + VISGUI_STATUS_COL +
  " , " + VXL_SHA1_COL +
  " , " + VXL_STATUS_COL +
  " , " + SECURITY_CLASSIFICATION_COL +
  " , " + SECURITY_POLICY_NAME_COL +
  " , " + SECURITY_CONTROL_SYSTEM_COL +
  " , " + SECURITY_DISSEMINATION_COL +
  " , " + SECURITY_RELEASABILITY_COL +
  " , " + SESSION_DESCRIPTION_COL +
  " , " + SESSION_MODIFIED_BY_COL +
  " , " + DATE_CREATED_COL +
  " ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ";

static const std::string INSERT_SESSION_METADATA =
  "INSERT INTO " + SESSION_METADATA_TABLE_NAME +
  " ( " + SESSION_AOI_ID_COL        +
  " , " + PARENT_SESSION_ID_COL    +
  " , " + SESSION_TYPE_COL         +
  " , " + IS_GROUND_TRUTH_COL         +
  " , " + PERSEAS_SHA1_COL +
  " , " + PERSEAS_STATUS_COL +
  " , " + APPS_SHA1_COL +
  " , " + APPS_STATUS_COL +
  " , " + DATA_SHA1_COL +
  " , " + DATA_STATUS_COL +
  " , " + QTTESTING_SHA1_COL +
  " , " + QTTESTING_STATUS_COL +
  " , " + VIDTK_SHA1_COL +
  " , " + VIDTK_STATUS_COL +
  " , " + VISGUI_SHA1_COL +
  " , " + VISGUI_STATUS_COL +
  " , " + VXL_SHA1_COL +
  " , " + VXL_STATUS_COL +
  " , " + SECURITY_CLASSIFICATION_COL +
  " , " + SECURITY_POLICY_NAME_COL +
  " , " + SECURITY_CONTROL_SYSTEM_COL +
  " , " + SECURITY_DISSEMINATION_COL +
  " , " + SECURITY_RELEASABILITY_COL +
  " , " + SESSION_DESCRIPTION_COL +
  " , " + SESSION_MODIFIED_BY_COL +
  " , " + DATE_CREATED_COL +
  " ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ";


static const std::string INSERT_SESSION_PARAMS =
  "INSERT INTO " + SESSION_PARAMS_TABLE_NAME +
  " ( " + PARAMS_SESSION_META_ID_COL  +
  " , " + PARAM_NAME_COL +
  " , " + PARAM_VALUE_COL +
  " ) VALUES (?,?,?);" ;

static const std::string SELECT_SESSION_METADATA =
  "SELECT " + SESSION_AOI_ID_COL +
  " , "     + PARENT_SESSION_ID_COL +
  " , "     + SESSION_TYPE_COL +
  " , "     + IS_GROUND_TRUTH_COL +
  " , "     + PERSEAS_SHA1_COL  +
  " , "     + PERSEAS_STATUS_COL +
  " , "     + APPS_SHA1_COL +
  " , "     + APPS_STATUS_COL +
  " , "     + DATA_SHA1_COL +
  " , "     + DATA_STATUS_COL +
  " , "     + QTTESTING_SHA1_COL +
  " , "     + QTTESTING_STATUS_COL +
  " , "     + VIDTK_SHA1_COL +
  " , "     + VIDTK_STATUS_COL +
  " , "     + VISGUI_SHA1_COL +
  " , "     + VISGUI_STATUS_COL +
  " , "     + VXL_SHA1_COL +
  " , "     + VXL_STATUS_COL +
  " , "     + SECURITY_CLASSIFICATION_COL +
  " , "     + SECURITY_POLICY_NAME_COL +
  " , "     + SECURITY_CONTROL_SYSTEM_COL +
  " , "     + SECURITY_DISSEMINATION_COL +
  " , "     + SECURITY_RELEASABILITY_COL +
  " , "     + SESSION_DESCRIPTION_COL  +
  " , "     + SESSION_MODIFIED_BY_COL +
  " , "     + DATE_CREATED_COL +
  " , "     + PARAM_NAME_COL +
  " , "     + PARAM_VALUE_COL +
  " FROM "  + SESSION_METADATA_TABLE_NAME  +
  " LEFT OUTER JOIN "  + SESSION_PARAMS_TABLE_NAME +
  " ON ( "  + SESSION_META_ID_PK_COL +
  " = "     + PARAMS_SESSION_META_ID_COL +
  " ) WHERE " + SESSION_META_ID_PK_COL +
  " =? ";

static const std::string SELECT_ALL_SESSION_IDS =
  "SELECT " + SESSION_META_ID_PK_COL +
  " , "     + DATE_CREATED_COL       +
  " , "     + PARENT_SESSION_ID_COL    +
  " , "     + SESSION_TYPE_COL    +
  " , "     + SESSION_AOI_ID_COL     +
  " FROM "  + SESSION_METADATA_TABLE_NAME +
  " ORDER BY " + SESSION_META_ID_PK_COL;


static const std::string SELECT_SESSION_PARENT_ID =
  " SELECT " + PARENT_SESSION_ID_COL +
  " FROM   " + SESSION_METADATA_TABLE_NAME +
  " WHERE  " + SESSION_META_ID_PK_COL      +
  " =? ";

static const std::string SELECT_MAX_SESSION_ID =
  " SELECT MAX( "  + SESSION_META_ID_PK_COL +
  " ) AS LAST_ID " +
  " FROM        "  + SESSION_METADATA_TABLE_NAME;

static const std::string SELECT_SESSION_PARAMS =
  "SELECT " + PARAM_NAME_COL  +
  ", "      + PARAM_VALUE_COL +
  " FROM "  + SESSION_PARAMS_TABLE_NAME  +
  " WHERE " + PARAMS_SESSION_META_ID_COL +
  " = ?";

static const std::string DELETE_SESSION_PARAMS =
  " DELETE FROM " + SESSION_PARAMS_TABLE_NAME  +
  " WHERE       " + PARAMS_SESSION_META_ID_COL +
  " =?";

static const std::string DELETE_SESSION =
  " DELETE FROM " + SESSION_METADATA_TABLE_NAME +
  " WHERE       " + SESSION_META_ID_PK_COL      +
  " =?";


}// namespace vidtk


#endif// __QUERY_MANAGER_H__
