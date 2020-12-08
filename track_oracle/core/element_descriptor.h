// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_ELEMENT_DESCRIPTOR_H
#define INCL_ELEMENT_DESCRIPTOR_H

#include <vital/vital_config.h>
#include <track_oracle/core/track_oracle_export.h>

#include <string>

//
// Everything you needed to know about a element type managed
// by track_oracle.
//
// Note that the "identity" of a data column is still determined
// only by the name and type.  element_role is only advisory.
// This is mostly important because one API for creating a element_store
// (aka data column) takes a complete element_descriptor; another API
// (mostly used for lookup) takes only the name and assumes the type
// will match.
//

namespace kwiver {
namespace track_oracle {

struct TRACK_ORACLE_EXPORT element_descriptor
{
  // is this element a system type (e.g. __frame_list),
  // a "well known" type meant to be shared across schemas (e.g. frame_number),
  // or an ad-hoc type the user added on the fly?
  enum element_role { INVALID, SYSTEM, WELLKNOWN, ADHOC };

  // the unique-across-track-oracle name of the field (e.g. "frame_number")
  std::string name;

  // a description of the field
  std::string description;

  // the machine-dependent typeid string of the type
  std::string typeid_str;

  // the role
  element_role role;

  // role helper functions
  static std::string role2str( element_role e );
  static element_role str2role ( const std::string& s );

  element_descriptor( const std::string& n,
                      const std::string& d,
                      const std::string& t,
                      element_role r ) :
    name(n), description(d), typeid_str(t), role(r)
  {}

  element_descriptor() :
    name("none"), description("none"), typeid_str(""), role(INVALID)
  {}

  bool is_valid() const { return role != INVALID; }

};

} // ...track_oracle
} // ...kwiver

#endif
