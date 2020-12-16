// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_LOGGING_MAP_H
#define INCL_LOGGING_MAP_H

///
/// Reduce logging verbosity.
///

#include <vital/vital_config.h>
#include <track_oracle/utils/logging_map_export.h>

#include <string>
#include <map>

#include <vital/logger/logger.h>
#include <vital/logger/location_info.h>

namespace kwiver
{
class LOGGING_MAP_EXPORT logging_map_type
{
public:
  logging_map_type( vital::logger_handle_t logger, const vital::logger_ns::location_info& s  );

  logging_map_type& set_output_prefix( const std::string& s );
  bool add_msg( const std::string& msg );
  bool empty() const;
  size_t n_msgs() const;
  void dump_msgs( vital::kwiver_logger::log_level_t level = vital::kwiver_logger::LEVEL_INFO ) const;
  void clear();

private:
  vital::logger_handle_t my_logger;
  vital::logger_ns::location_info site;
  std::string output_prefix;
  std::map< std::string, std::size_t > msg_map;
};

} // kwiver

#endif
