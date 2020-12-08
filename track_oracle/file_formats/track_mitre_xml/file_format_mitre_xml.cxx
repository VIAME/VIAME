// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "file_format_mitre_xml.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <tinyxml.h>

#include <track_oracle/utils/tokenizers.h>

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

using std::string;
using std::vector;

namespace kwiver {
namespace track_oracle {

bool
file_format_mitre_xml
::inspect_file( const string& fn ) const
{
  vector< string > tokens = xml_tokenizer::first_n_tokens( fn, 10 );
  for (size_t i=0; i<tokens.size(); ++i)
  {
    if (tokens[i].find( "<queryRegionElements>" ) != string::npos ) return true;
  }
  return false;
}

bool
file_format_mitre_xml
::read( const string& fn,
        track_handle_list_type& tracks ) const
{
  track_mitre_xml_type mitre_xml;

  TiXmlDocument doc( fn.c_str() );
  if ( ! doc.LoadFile() )
  {
    LOG_ERROR( main_logger, "MITRE XML file '" << fn << "': TinyXML failed to load document");
    return false;
  }

  TiXmlNode* xmlRoot = doc.RootElement();
  if ( ! xmlRoot )
  {
    LOG_INFO( main_logger, "MITRE XML file '" << fn << "': TinyXML found no root node?");
    return false;
  }

  TiXmlNode* xmlTrackObjects = 0;
  bool first_loop = true;
  while( (xmlTrackObjects = xmlRoot->IterateChildren( xmlTrackObjects )) )
  {
    TiXmlElement* e = xmlTrackObjects->ToElement();
    if (!e)
    {
      LOG_ERROR( main_logger, "MITRE XML file '" << fn << "': TinyXML couldn't cast to element?");
      return false;
    }

    int frameNumber;
    double x, y, width, height;

    // Get the data from the element...
    if (e->QueryIntAttribute( "frameNumber", &frameNumber) != TIXML_SUCCESS)
    {
      LOG_INFO( main_logger, "MITRE XML file '" << fn << "': row " << e->Row() << ": no frameNumber?");
      return false;
    }
    if (e->QueryDoubleAttribute( "height", &height ) != TIXML_SUCCESS )
    {
      LOG_INFO( main_logger, "MITRE XML file '" << fn << "': row " << e->Row() << ": no height?");
      return false;
      }
    if (e->QueryDoubleAttribute( "width", &width ) != TIXML_SUCCESS )
    {
      LOG_INFO( main_logger, "MITRE XML file '" << fn << "': row " << e->Row() << ": no width?");
      return false;
    }
    if (e->QueryDoubleAttribute( "x", &x ) != TIXML_SUCCESS )
    {
      LOG_INFO( main_logger, "MITRE XML file '" << fn << "': row " << e->Row() << ": no x?");
      return false;
    }
    if (e->QueryDoubleAttribute( "y", &y ) != TIXML_SUCCESS )
    {
      LOG_INFO( main_logger, "MITRE XML file '" << fn << "': row " << e->Row() << ": no y?");
      return false;
    }

    if (first_loop)
    {
      // only one track per file
      tracks.push_back( mitre_xml.create() );
    }

    frame_handle_type f = mitre_xml.create_frame();
    mitre_xml[ f ].frame_number() = frameNumber;
    mitre_xml[ f ].bounding_box() =
      vgl_box_2d<double>(
        vgl_point_2d<double>( x, y ),
        vgl_point_2d<double>( x+width, y+height ));
    first_loop = false;

  } // ...while more frames

  // all done!
  return true;

}

} // ...track_oracle
} // ...kwiver
