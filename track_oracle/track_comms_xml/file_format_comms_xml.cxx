/*ckwg +5
 * Copyright 2012-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "file_format_comms_xml.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <algorithm>
#include <iterator>

#include <tinyxml.h>

#include <track_oracle/utils/tokenizers.h>
#include <vgl/vgl_point_2d.h>

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

using std::back_inserter;
using std::copy;
using std::find;
using std::reverse;
using std::string;
using std::vector;

namespace // anon
{

bool
load_tracks( const string& filename,
             const string& query_id,
             ::kwiver::track_oracle::track_comms_xml_type& comms_xml,
             TiXmlNode* activitySet,
             ::kwiver::track_oracle::track_handle_list_type& track_ids )
{
  TiXmlNode* xmlTrackObjects = 0;
  while( (xmlTrackObjects = activitySet->IterateChildren( xmlTrackObjects )) )
  {
    TiXmlElement* e = xmlTrackObjects->ToElement();
    if (!e)
    {
      LOG_ERROR( main_logger, "COMMS XML file '" << filename << "': TinyXML couldn't cast to element?" );
      return false;
    }

    if ( strcmp( e->Value(), "Activity" ) )
    {
      continue;
    }

    string track_source;
    string track_source_file;
    double prob = 1.0;

    if ( e->QueryStringAttribute( "strmid", &track_source ) != TIXML_SUCCESS )
    {
      LOG_ERROR( main_logger, "COMMS XML file '" << filename << "': row " << e->Row() << ": no strmid?" );
      return false;
    }

    if ( e->QueryValueAttribute( "sim", &prob ) != TIXML_SUCCESS )
    {
      LOG_WARN( main_logger, "COMMS XML file '" << filename << "': row " << e->Row() << ": no prob? ");
    }

    // Can't use vul_file here because we always use POSIX directory separators.
    string::reverse_iterator sep = find( track_source.rbegin(), track_source.rend(), '/' );
    copy( track_source.rbegin(), sep, back_inserter( track_source_file ) );
    reverse( track_source_file.begin(), track_source_file.end() );

    ::kwiver::track_oracle::track_handle_type t = comms_xml.create();
    track_ids.push_back( t );

    comms_xml.track_source() = track_source_file;
    comms_xml.probability() = prob;
    comms_xml.query_id() = query_id;

    TiXmlNode* xmlFrameObjects = 0;
    while( (xmlFrameObjects = e->IterateChildren( xmlFrameObjects )) )
    {
      TiXmlElement* f = xmlFrameObjects->ToElement();
      if (!f)
      {
        LOG_ERROR( main_logger, "COMMS XML file '" << filename << "': TinyXML couldn't cast to element?" );
        return false;
      }

      unsigned long long frameTime;
      double x1, y1, x2, y2;

      // Get the data from the element...
      if (f->QueryValueAttribute( "frame", &frameTime) != TIXML_SUCCESS)
      {
        LOG_ERROR( main_logger, "COMMS XML file '" << filename << "': row " << f->Row() << ": no frame?" );
        return false;
      }
      if (f->QueryValueAttribute( "xmin", &x1 ) != TIXML_SUCCESS )
      {
        LOG_ERROR( main_logger, "COMMS XML file '" << filename << "': row " << f->Row() << ": no xmin?" );
        return false;
      }
      if (f->QueryValueAttribute( "ymin", &y1 ) != TIXML_SUCCESS )
      {
        LOG_ERROR( main_logger, "COMMS XML file '" << filename << "': row " << f->Row() << ": no ymin?" );
        return false;
      }
      if (f->QueryValueAttribute( "xmax", &x2 ) != TIXML_SUCCESS )
      {
        LOG_ERROR( main_logger, "COMMS XML file '" << filename << "': row " << f->Row() << ": no xmax?" );
        return false;
      }
      if (f->QueryValueAttribute( "ymax", &y2 ) != TIXML_SUCCESS )
      {
        LOG_ERROR( main_logger, "COMMS XML file '" << filename << "': row " << f->Row() << ": no ymax?" );
        return false;
      }

      ::kwiver::track_oracle::frame_handle_type h = comms_xml.create_frame();
      comms_xml[ h ].timestamp() = 1000 * frameTime;
      comms_xml[ h ].bounding_box() =
        vgl_box_2d<double>(
          vgl_point_2d<double>( x1, y1 ),
          vgl_point_2d<double>( x2, y2 ));
    }

  } // ...while more frames

  return true;
}

} // ...anon

namespace kwiver {
namespace track_oracle {

comms_xml_reader_opts&
comms_xml_reader_opts
::operator=( const file_format_reader_opts_base& rhs_base )
{
  const comms_xml_reader_opts* rhs = dynamic_cast<const comms_xml_reader_opts*>(&rhs_base);

  if (rhs)
  {
    this->set_comms_qid( rhs->comms_qid );
  }
  else
  {
    LOG_WARN( main_logger,"Assigned a non-comms_xml options structure to a comms_xml options structure: Slicing the class");
  }

  return *this;
}

bool
file_format_comms_xml
::inspect_file( const string& fn ) const
{
  vector< string > tokens = xml_tokenizer::first_n_tokens( fn, 10 );
  for (size_t i=0; i<tokens.size(); ++i)
  {
    if (tokens[i].find( "<thRecv" ) != string::npos) return true;
  }
  return false;
}

bool
file_format_comms_xml
::read( const string& fn,
        track_handle_list_type& tracks ) const
{
  track_comms_xml_type comms_xml;

  TiXmlDocument doc( fn.c_str() );
  if ( ! doc.LoadFile() )
  {
    LOG_ERROR( main_logger, "COMMS XML file '" << fn << "': TinyXML failed to load document");
    return false;
  }

  TiXmlNode* thRecv = 0;
  while( (thRecv = doc.IterateChildren( "thRecv", thRecv )) )
  {
    TiXmlNode* r = thRecv->FirstChild( "VIRATQueryResp" );
    if (!r) continue;

    TiXmlElement* e = r->ToElement();
    if (!e)
    {
      LOG_ERROR( main_logger, "COMMS XML file '" << fn << "': TinyXML couldn't cast to element?");
      return false;
    }

    string mtype;
    if ( e->QueryValueAttribute( "mtype", &mtype ) != TIXML_SUCCESS ) continue;
    if ( mtype != "final" ) continue;

    string resp_qid;
    if ( e->QueryValueAttribute( "qid", &resp_qid ) != TIXML_SUCCESS ) continue;
    if ( ! this->opts.comms_qid.empty() && ( resp_qid != this->opts.comms_qid )) continue;

    TiXmlNode* activitySet = r->FirstChild( "ActivitySet" );
    if ( ! activitySet ) continue;

    if ( ! load_tracks( fn, resp_qid, comms_xml, activitySet, tracks ) )
    {
      return false;

    }
  }

  // all done!
  return true;

}

} // ...track_oracle
} // ...kwiver
