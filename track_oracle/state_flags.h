/*ckwg +5
 * Copyright 2014-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef INCL_STATE_FLAGS_H
#define INCL_STATE_FLAGS_H

///
/// These are a set of flags (one each for frames, tracks, events)
/// which can be applied at various stages of the scoring pipeline
/// to record events such as filtering on AOI, filtering on time,
/// matched / unmatched, etc.
///
///
/// The default format is 'key:value'; a key may have a single value
/// at any time.  To maintain compatibility with the old CSV flags_t
/// type, 'key' is also allowed with an implicit value of "", which
/// is different from the key being absent.
///
/// Conceptually, this is a map of string->string; for example,
/// 'origin'->'truth' or 'origin'->'computed'.  Call the first string
/// the component and the second string the status.  Each component
/// has a single status.
///
/// Now, storing this as a map of strings would be hugely wasteful.
/// Instead, we store it as a vector of ints, where the index indicates
/// the component and the value indicates the status.
///
/// So where do we store the lookup table for the component and status
/// strings?  In a singleton sidebar map.  As long as any persistent
/// data is read into and out of the map via the data term (i.e. as
/// strings), the numeric values in the data term are irrelevant from
/// session to session.
///
///
///

#include <vital/vital_config.h>
#include <track_oracle/track_oracle_export.h>

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <track_oracle/data_terms/data_terms_common.h>
#include <track_oracle/kwiver_io_base.h>

class TiXmlElement;

namespace kwiver {
namespace track_oracle {

struct TRACK_ORACLE_EXPORT state_flag_type
{
  void set_flag( const std::string& component, const std::string& status = "" );
  void clear_flag( const std::string& component );
  std::map< std::string, std::string > get_flags() const;
  bool operator==( const state_flag_type& rhs ) const;

private:
  std::vector<size_t> data;
};

std::ostream& TRACK_ORACLE_EXPORT operator<<( std::ostream& os, const state_flag_type& t );
std::istream& TRACK_ORACLE_EXPORT operator>>( std::istream& os, state_flag_type& t );


namespace dt {
namespace utility {

struct TRACK_ORACLE_EXPORT state_flags: public data_term_base, kwiver_io_base<state_flag_type>
{
  state_flags(): kwiver_io_base< state_flag_type >( "attributes" ) {}
  typedef state_flag_type Type;
  static context c;
  static std::string get_context_name() { return "attributes"; }
  static std::string get_context_description() { return "domain-defined attribute flags"; }
};

} // ...utility
} // ...dt

} // ...track_oracle
} // ...kwiver

#endif
