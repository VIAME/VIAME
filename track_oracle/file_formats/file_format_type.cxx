// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "file_format_type.h"

#include <ostream>
#include <track_oracle/core/track_oracle_api_types.h>
#include <track_oracle/core/element_store.h>
#include <track_oracle/file_formats/file_format_manager.h>

using std::map;
using std::ostream;
using std::string;

namespace kwiver {
namespace track_oracle {

string
file_format_type
::to_string( file_format_enum f )
{
  switch (f)
  {
  case TF_KW18:           return "kw18";
  case TF_XGTF:           return "xgtf";
  case TF_KWXML:          return "kwxml";
  case TF_APIX:           return "apix";
  case TF_MITRE_BOX_XML:  return "mitre-box-xml";
  case TF_COMMS_XML:      return "comms-xml";
  case TF_KST:            return "kst";
  case TF_VATIC:          return "vatic";
  case TF_VPD_TRACK:      return "vpd-track";
  case TF_VPD_EVENT:      return "vpd-event";
  case TF_E2AT_CALLOUT:   return "e2at-callout";
  case TF_4676:           return "4676";
  case TF_CSV:            return "csv";
  case TF_KWIVER:         return "kwiver";
  case TF_KPF_GEOM:       return "kpf-geom";
  case TF_KPF_ACT:       return "kpf-act";
  case TF_INVALID_TYPE:   return "invalid";
  }
  return "invalid";
}

file_format_enum
file_format_type
::from_string( const string& s )
{
  if      ( s == file_format_type::to_string( TF_KW18 ))  return TF_KW18;
  else if ( s == file_format_type::to_string( TF_XGTF ))  return TF_XGTF;
  else if ( s == file_format_type::to_string( TF_KWXML )) return TF_KWXML;
  else if ( s == file_format_type::to_string( TF_APIX ))  return TF_APIX;
  else if ( s == file_format_type::to_string( TF_MITRE_BOX_XML ))  return TF_MITRE_BOX_XML;
  else if ( s == file_format_type::to_string( TF_COMMS_XML ))  return TF_COMMS_XML;
  else if ( s == file_format_type::to_string( TF_KST ))    return TF_KST;
  else if ( s == file_format_type::to_string( TF_VATIC ))  return TF_VATIC;
  else if ( s == file_format_type::to_string( TF_VPD_TRACK ))  return TF_VPD_TRACK;
  else if ( s == file_format_type::to_string( TF_VPD_EVENT ))  return TF_VPD_EVENT;
  else if ( s == file_format_type::to_string( TF_E2AT_CALLOUT )) return TF_E2AT_CALLOUT;
  else if ( s == file_format_type::to_string( TF_4676 ))   return TF_4676;
  else if ( s == file_format_type::to_string( TF_CSV ))    return TF_CSV;
  else if ( s == file_format_type::to_string( TF_KWIVER ))  return TF_KWIVER;
  else if ( s == file_format_type::to_string( TF_KPF_GEOM ))    return TF_KPF_GEOM;
  else if ( s == file_format_type::to_string( TF_KPF_ACT ))    return TF_KPF_ACT;
  else return TF_INVALID_TYPE;
}

} // ...track_oracle
} // ...kwiver
