// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_AOI_UTILS_H
#define INCL_AOI_UTILS_H

///
/// Track oracle's AOI class.  Can be either pixel or lat/lon.
/// See help_text() for the format of the construction string.
///
/// ***
/// *** THIS CLASS WILL THROW an aoi_exception if the format string
/// *** is incorrect!  Also if the lat/lon AOI doesn't fit in a single
/// *** MGRS zone.
/// ***
///

#include <vital/vital_config.h>
#include <track_oracle/aoi_utils/track_oracle_aoi_utils_export.h>

#include <string>
#include <stdexcept>

namespace kwiver {
namespace track_oracle {

namespace aoi_utils {

struct aoi_impl;

class TRACK_ORACLE_AOI_UTILS_EXPORT aoi_t
{
public:
  enum flavor_t { INVALID, PIXEL, GEO };

  aoi_t();
  explicit aoi_t( const std::string& s );   // can throw!
  void set( const std::string& s );       // can throw!
  virtual ~aoi_t();

  flavor_t flavor() const;
  bool in_aoi( double x, double y) const;

  std::string to_str() const;

  static std::string help_text();

private:
  aoi_impl* p;
};

class TRACK_ORACLE_AOI_UTILS_EXPORT aoi_exception: public std::runtime_error
{
public:
  explicit aoi_exception( const std::string& s ): std::runtime_error( s ) {}
};

} // ...aoi_utils
} // ...track_oracle
} // ...kwiver

#endif
