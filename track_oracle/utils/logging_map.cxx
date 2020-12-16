// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "logging_map.h"
#include <sstream>

using std::string;
using std::ostringstream;
using std::map;

using kwiver::vital::kwiver_logger;

namespace kwiver
{

logging_map_type
::logging_map_type( vital::logger_handle_t logger, const vital::logger_ns::location_info& s )
  : my_logger(logger), site( s ), output_prefix( "" )
{
}

logging_map_type&
logging_map_type
::set_output_prefix( const string& s )
{
  this->output_prefix = s;
  return *this;
}

bool
logging_map_type
::add_msg( const string& msg )
{
  return (msg_map[ msg ]++ == 0);
}

bool
logging_map_type
::empty() const
{
  return msg_map.empty();
}

size_t
logging_map_type
::n_msgs() const
{
  return msg_map.size();
}

void
logging_map_type
::dump_msgs( kwiver_logger::log_level_t level ) const
{
  for (map<string, size_t>::const_iterator i=this->msg_map.begin(), e=this->msg_map.end();
       i != e;
       ++i)
  {
    ostringstream oss;
    oss << this->output_prefix << i->first << " : count " << i ->second;

    switch( level )
    {
    case kwiver_logger::LEVEL_TRACE:
      my_logger->log_trace( oss.str(), this->site ); break;
    case kwiver_logger::LEVEL_DEBUG:
      my_logger->log_debug( oss.str(), this->site ); break;
    case kwiver_logger::LEVEL_INFO:
      my_logger->log_info( oss.str(), this->site ); break;
    case kwiver_logger::LEVEL_WARN:
      my_logger->log_warn( oss.str(), this->site ); break;
    case kwiver_logger::LEVEL_ERROR:
      my_logger->log_error( oss.str(), this->site ); break;
    case kwiver_logger::LEVEL_FATAL:
      my_logger->log_fatal( oss.str(), this->site ); break;
    default:
      my_logger->log_info( oss.str(), this->site ); break;
    }
  }
};

void
logging_map_type
::clear()
{
  this->msg_map.clear();
}

} // vidtk
