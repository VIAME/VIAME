/*ckwg +5
 * Copyright 2014-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef INCL_KWIVER_IO_HELPERS
#define INCL_KWIVER_IO_HELPERS

#include <vital/vital_config.h>
#include <track_oracle/track_oracle_export.h>

#include <iostream>
#include <vector>
#include <string>

#include <vgl/vgl_point_2d.h>
#include <vgl/vgl_box_2d.h>
#include <vgl/vgl_point_3d.h>
#include <vital/types/timestamp.h>

namespace kwiver {
namespace track_oracle {

std::ostream& TRACK_ORACLE_EXPORT kwiver_write_highprecision( std::ostream& os, double d, std::streamsize new_prec = 10 );

bool TRACK_ORACLE_EXPORT kwiver_read( const std::string& s, vgl_point_2d<double>& d );
std::ostream& TRACK_ORACLE_EXPORT kwiver_write( std::ostream& os, const vgl_point_2d<double>& d, const std::string& sep );
std::vector<std::string> TRACK_ORACLE_EXPORT kwiver_csv_header_pair( const std::string& n, const std::string& p1, const std::string& p2 );

bool TRACK_ORACLE_EXPORT kwiver_read( const std::string& s, vgl_box_2d<double>& d );
std::ostream& TRACK_ORACLE_EXPORT kwiver_write( std::ostream& os, const vgl_box_2d<double>& d, const std::string& sep );
std::vector<std::string> TRACK_ORACLE_EXPORT kwiver_box_2d_headers( const std::string& s );

bool TRACK_ORACLE_EXPORT kwiver_read( const std::string& s, vgl_point_3d<double>& d );
std::ostream& TRACK_ORACLE_EXPORT kwiver_write( std::ostream& os, const vgl_point_3d<double>& d, const std::string& sep );
std::vector<std::string> TRACK_ORACLE_EXPORT kwiver_point_3d_headers( const std::string& n );

std::pair<std::string, std::string > TRACK_ORACLE_EXPORT kwiver_ts_to_strings( const vital::timestamp& ts );
bool TRACK_ORACLE_EXPORT kwiver_ts_string_read( const std::string& frame_str,
                                                const std::string& time_str,
                                                vital::timestamp& t );
bool TRACK_ORACLE_EXPORT kwiver_read( const std::string& s, vital::timestamp& ts );
std::ostream& TRACK_ORACLE_EXPORT kwiver_write( std::ostream& os, const vital::timestamp& ts );

//
// default, unimplemented output routine for TMP
//

char TRACK_ORACLE_EXPORT kwiver_write( std::ostream& os, ... );

} // ...track_oracle
} // ...kwiver

#endif
