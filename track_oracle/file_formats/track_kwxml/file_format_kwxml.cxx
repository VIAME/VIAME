/*ckwg +5
 * Copyright 2012-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "file_format_kwxml.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <utility>

#include <tinyxml.h>

#include <track_oracle/utils/tokenizers.h>
#include <track_oracle/utils/logging_map.h>
#include <track_oracle/aries_interface/aries_interface.h>

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );


using std::istringstream;
using std::map;
using std::ofstream;
using std::ostream;
using std::string;
using std::vector;

namespace // anon
{


template< typename T>
bool
read_vector_from_xml( TiXmlElement* descriptor,
                      const string& tag,
                      vector<T>& d )
{
  TiXmlElement* e = descriptor->FirstChildElement( tag );
  if ( ! e )
  {
    LOG_ERROR( main_logger,"No " << tag << " data at " << descriptor->Row() << "?\n");
    return false;
  }
  istringstream iss( e->Attribute( "value" ));
  T tmp;
  while ( (iss >> tmp ))
  {
    d.push_back( tmp );
  }
  return true;
}

} // anon


namespace kwiver {
namespace track_oracle {

kwxml_reader_opts&
kwxml_reader_opts
::operator=( const file_format_reader_opts_base& rhs_base )
{
  const kwxml_reader_opts* rhs = dynamic_cast<const kwxml_reader_opts*>(&rhs_base);

  if (rhs)
  {
    this->set_track_style_filter( rhs->track_style_filter );
  }
  else
  {
    LOG_WARN( main_logger,"Assigned a non-kwxml options structure to a kwxml options structure: Slicing the class");
  }

  return *this;
}

bool
file_format_kwxml
::inspect_file( const string& fn ) const
{
  vector< string > tokens = xml_tokenizer::first_n_tokens( fn, 1 );
  for (size_t i=0; i<tokens.size(); ++i)
  {
    if (tokens[i].find( "<vibrantDescriptors>" ) != string::npos) return true;
  }
  return false;
}

bool
file_format_kwxml
::read( const string& fn,
        track_handle_list_type& tracks ) const
{
  // dig through the XML wrappers...

  LOG_INFO( main_logger, "TinyXML loading '" << fn << "': start" );
  TiXmlDocument doc( fn.c_str() );
  TiXmlHandle doc_handle( &doc );
  if ( ! doc.LoadFile() )
  {
    LOG_ERROR( main_logger,"TinyXML (KWXML) couldn't load '" << fn << "'; skipping\n");
    return false;
  }
  LOG_INFO( main_logger, "TinyXML loading '" << fn << "': complete" );

  TiXmlNode* xml_root = doc.RootElement();
  if ( ! xml_root )
  {
    LOG_ERROR( main_logger,"Couldn't load root element from '" << fn << "'; skipping\n");
    return false;
  }
  // The queryResults.xml files don't have a root node, and just have
  // a series of "track" nodes. Luckily, tinyXML will happily deal
  // with these malformed "xml" files: we just have to treat the
  // document as the root node.
  if ( string("track") == xml_root->Value() )
  {
    xml_root = &doc;
  }


  TiXmlNode* xml_track_objects = 0;
  track_kwxml_type kwxml;
  logging_map_type wmap( main_logger, KWIVER_LOGGER_SITE );

  while( (xml_track_objects = xml_root->IterateChildren( xml_track_objects )) )
  {
    // only interested in track nodes
    const string object_str("track");
    if (xml_track_objects->Value() != object_str) continue;


    // The frameNumberOrigin and offset variables are deprecated and no longer
    // used. Therefore, we won go through the problem of reading them.

    TiXmlElement* top_e = xml_track_objects->ToElement();
    if (!top_e)
    {
      LOG_ERROR( main_logger,"Couldn't cast track to element?\n");
      return false;
    }

    TiXmlHandle track_handle( top_e );

    //
    // treat track style a little differently, since we may want to skip everything else
    // if we filter this track out based on it
    //

    string track_style_str;
    {
      TiXmlText* track_style_text = track_handle.FirstChildElement( "trackStyle" ).FirstChild().ToText();
      if( track_style_text )
      {
        track_style_str = track_style_text->ValueStr();
      }
    }

    // don't create unless it passes the filter
    if ( ( ! this->opts.track_style_filter.empty() ) &&
         ( track_style_str != this->opts.track_style_filter ))
    {
      continue;
    }

    // Create track in the cloud!
    tracks.push_back(kwxml.create());

    // Set the track style
    kwxml.track_style() = track_style_str;

    //
    // begin e2at support
    //

    {
      TiXmlText* txt = track_handle.FirstChildElement( "clipFilename" ).FirstChild().ToText();
      if ( txt ) kwxml.clip_filename() = txt->ValueStr();
    }
    {
      TiXmlText* txt = track_handle.FirstChildElement( "basicAnnotation" ).FirstChild().ToText();
      if ( txt ) kwxml.basic_annotation() = txt->ValueStr();
    }
    {
      TiXmlText* txt = track_handle.FirstChildElement( "augmentedAnnotation" ).FirstChild().ToText();
      if ( txt ) kwxml.augmented_annotation() = txt->ValueStr();
    }
    {
      TiXmlText* txt = track_handle.FirstChildElement( "start_time_secs" ).FirstChild().ToText();
      if ( txt )
      {
        istringstream iss( txt->ValueStr() );
        double tmp;
        if ( iss >> tmp ) kwxml.start_time_secs() = tmp;
      }
    }
    {
      TiXmlText* txt = track_handle.FirstChildElement( "end_time_secs" ).FirstChild().ToText();
      if ( txt )
      {
        istringstream iss( txt->ValueStr() );
        double tmp;
        if ( iss >> tmp ) kwxml.end_time_secs() = tmp;
      }
    }
    {
      TiXmlText* txt = track_handle.FirstChildElement( "latitude" ).FirstChild().ToText();
      if ( txt )
      {
        istringstream iss( txt->ValueStr() );
        double tmp;
        if ( iss >> tmp ) kwxml.latitude() = tmp;
      }
    }
    {
      TiXmlText* txt = track_handle.FirstChildElement( "longitude" ).FirstChild().ToText();
      if ( txt )
      {
        istringstream iss( txt->ValueStr() );
        double tmp;
        if ( iss >> tmp ) kwxml.longitude() = tmp;
      }
    }

    //
    // end e2at support
    //

    int external_id;
    if (top_e->QueryIntAttribute( "id", &external_id) == TIXML_SUCCESS)
    {
      kwxml.external_id() = external_id  ;
    }
    int frame;

    {
      unsigned video_id;
      TiXmlText* video_id_text = track_handle.FirstChildElement( "videoID" ).FirstChild().ToText();
      if( video_id_text )
      {
        istringstream istr( video_id_text->ValueStr() );
        istr >> video_id;
        kwxml.video_id() = video_id;
      }
      else
      {
        wmap.add_msg( "no VideoID?" );
      }
    }


    {
      TiXmlText* time_stamp_text = track_handle.FirstChildElement( "timeStamp" ).FirstChild().ToText();
      if( time_stamp_text )
      {
        kwxml.time_stamp() = time_stamp_text->ValueStr();
      }
      else
      {
        wmap.add_msg( "no track-level timestamp?" );
      }
    }

    {
      vector< unsigned > source_track_ids;
      TiXmlText* source_tracks_text = track_handle.FirstChildElement( "sourceTrackIDs" ).FirstChild().ToText();
      if( source_tracks_text )
      {
        istringstream istr( source_tracks_text->ValueStr() );
        unsigned id;
        while( istr >> id )
        {
          source_track_ids.push_back( id );
        }
        kwxml.source_track_ids() = source_track_ids;
      }
      else
      {
        wmap.add_msg( "no sourceTrackIDs?" );
      }
    }

    //Object assessment element not considered currently. If necessary, add here.

    //Frame information
    for ( TiXmlElement* bbox_child = xml_track_objects->FirstChildElement( "bbox");
          bbox_child != 0;
          bbox_child = bbox_child->NextSiblingElement( "bbox" ) )
    {

      frame_handle_type current_frame = kwxml.create_frame();
      string box_type, track_style;
      double ulx, uly, lrx, lry;
      unsigned long long timestamp = 0;

      if(bbox_child->QueryStringAttribute( "type", &box_type) != TIXML_SUCCESS )
      {
        wmap.add_msg("No boxType?");
      }

      if (bbox_child->QueryDoubleAttribute( "ulx", &ulx ) != TIXML_SUCCESS )
      {
        wmap.add_msg("No ULX?");
      }
      if (bbox_child->QueryDoubleAttribute( "uly", &uly ) != TIXML_SUCCESS )
      {
        wmap.add_msg("No ULY?");
      }
      if (bbox_child->QueryDoubleAttribute( "lrx", &lrx ) != TIXML_SUCCESS )
      {
        wmap.add_msg("No LRX?");
      }
      if (bbox_child->QueryDoubleAttribute( "lry", &lry ) != TIXML_SUCCESS )
      {
        wmap.add_msg("No LRY?");
      }
      if (bbox_child->QueryIntAttribute( "frame", &frame)  != TIXML_SUCCESS )
      {
        wmap.add_msg("No frame");
      }

      const char* timestamp_str = bbox_child->Attribute( "timestamp" );
      if ( timestamp_str )
      {
        istringstream iss( timestamp_str );

        if( !(iss >> timestamp) )
        {
          LOG_WARN( main_logger,"XML frame " << frame << ": couldn't parse timestamp string '" << timestamp_str << "' as unsigned long long\n");
        }
      }

      kwxml[current_frame].frame_number() = frame;
      kwxml[current_frame].type() = box_type;
      kwxml[current_frame].bounding_box() =
        vgl_box_2d<double>(
          vgl_point_2d<double>(ulx, uly),
          vgl_point_2d<double>(lrx, lry));
      kwxml[current_frame].timestamp_usecs() = timestamp;

    }

    // labels are (for the moment) not 'descriptors'

    {
      TiXmlElement* labels_node = xml_track_objects->FirstChildElement( "labels" );
      if (labels_node)
      {
        const char* label_domain_ptr = labels_node->Attribute( "domain" );
        if ( label_domain_ptr )
        {
          string label_domain( label_domain_ptr );
          if ( label_domain == "virat" )
          {
            descriptor_event_label_type delt;
            for (TiXmlElement* d = labels_node->FirstChildElement( "event" );
                 d != 0;
                 d = d->NextSiblingElement( "event" ))
            {
              single_event_label_type s;
              bool good = true;
              if ( d->QueryStringAttribute( "type", &s.activity_name ) != TIXML_SUCCESS )
              {
                LOG_WARN( main_logger, "labels node event sub-node without event-type attribute at " << d->Row() );
                good = false;
              }
              else
              {
                try
                {
                  aries_interface::activity_to_index( s.activity_name );
                }
                catch (aries_interface_exception&)
                {
                  LOG_WARN( main_logger, "labels node event '" << s.activity_name << "' is not a virat event at " << d->Row() );
                  good = false;
                }
              }
              if ( d->QueryDoubleAttribute( "spatialOverlap", &s.spatial_overlap ) != TIXML_SUCCESS )
              {
                LOG_WARN( main_logger, "labels node event sub-node without spatialOverlap attribute at " << d->Row() );
                good = false;
              }
              if ( d->QueryDoubleAttribute( "temporalOverlap", &s.temporal_overlap ) != TIXML_SUCCESS )
              {
                LOG_WARN( main_logger, "labels node event sub-node without temporalOverlap attribute at " << d->Row() );
                good = false;
              }

              if (good )
              {
                delt.labels.push_back( s );
              }
            } // .. all events

            kwxml.descriptor_event_label() = delt;

          } // virat domain
          else
          {
            LOG_WARN( main_logger, "labels node with unsupported domain '" << label_domain
                      << "' (only virat currently support); skipped at " << labels_node->Row() );
          }
        }
        else
        {
          LOG_WARN( main_logger, "labels node without domain attribute; skipped at " << labels_node->Row() );
        }
      } // ...if d
    } // ... event labels

    for (TiXmlElement* descriptor = xml_track_objects->FirstChildElement( "descriptor" );
         descriptor != 0;
         descriptor = descriptor->NextSiblingElement( "descriptor" ) )
    {
      const char* descr_type = descriptor->Attribute( "type" );

      if(!descr_type)
      {
        LOG_WARN( main_logger,"Descriptor element without type attribute at " << descriptor->Row() << "\n");
      }

      //Classifier descriptor not currently implemented, but should be added to this loop when done.
      if( descr_type && ( (string(descr_type)=="classifier" )
                          ||
                          (string(descr_type)=="PersonOrVehicleMovement") ))
      {
        vector< double > activity_probabilities;
        activity_probabilities.resize( aries_interface::index_to_activity_map().size(), 0.0 );

        for ( TiXmlElement* probNode = descriptor->FirstChildElement( "probability" );
              probNode != 0;
              probNode = probNode->NextSiblingElement( "probability" ) )
        {
          size_t activity_idx;
          try
          {
            activity_idx = aries_interface::activity_to_index( probNode->Attribute( "activity" ));
          }
          catch (aries_interface_exception& /*e*/)
          {
            LOG_ERROR( main_logger, "Couldn't recognize " << probNode->Attribute("activity")
                     << " as a valid activity?");
            return false;
          }
          double prob;
          if (probNode->QueryDoubleAttribute( "value", &prob )  != TIXML_SUCCESS )
          {
            LOG_ERROR( main_logger, "Couldn't find a probability at " << probNode->Row() << "?");
            prob = 0;
          }
          /// Old files could have a value of "-1" written to the
          /// XML. These should be ignored.
          if( prob != -1 )
          {
            activity_probabilities[ activity_idx ] = prob;
          }
        }
        kwxml.descriptor_classifier() = activity_probabilities;
      }

      else if (string(descr_type) == "texasHOF" )
      {
        int n_rows = 0;
        vector< vector< double > > texas_hof_vectors;
        TiXmlElement* count_e = descriptor->FirstChildElement( "vectorHOFCount" );
        if ( (!count_e) || (count_e->QueryIntAttribute( "value", &n_rows) != TIXML_SUCCESS))
        {
          LOG_WARN( main_logger,"No vectorHOFCount at " << descriptor->Row() << "?\n");
        }
        for (unsigned i=0; i<static_cast<unsigned>(n_rows); i++)
        {
          texas_hof_vectors.push_back( vector<double>() );
        }

        TiXmlNode* xml_row_node = 0;
        while ( (xml_row_node = descriptor->IterateChildren( xml_row_node )))
        {
          TiXmlElement* e = xml_row_node->ToElement();
          if (!e)
          {
            LOG_WARN( main_logger,"Couldn't convert to element at " << xml_row_node->Row() << "?\n");
          }
          const string vector_hof_str("vectorHOF");
          if (xml_row_node->Value() != vector_hof_str) continue;

          int index;
          if (e->QueryIntAttribute( "id", &index ) != TIXML_SUCCESS)
          {
            LOG_WARN( main_logger,"Couldn't parse id near " << xml_row_node->Row() << "?\n");
          }
          vector<double>& hof_row = texas_hof_vectors[index];

          istringstream iss( e->Attribute( "value" ) );
          double tmp;
          while ( (iss >> tmp))
          {
            hof_row.push_back( tmp );
          }
        }

        kwxml.descriptor_uthof() = texas_hof_vectors;
      }

      else if (string(descr_type) == "rpiDBN1" )
      {
        vector< double > dbn1_data;
        if ( ! read_vector_from_xml<double>( descriptor, "vectorRPIDBN1", dbn1_data ) )
        {
          LOG_WARN( main_logger,"No rpiDBN1 data at " << descriptor->Row() << "\n");
        }
        else
        {
          kwxml.descriptor_rpidbn1() = dbn1_data;
        }
      }

      else if (string(descr_type) == "rpiDBN2" )
      {
        vector< double > dbn2_data;
        if ( ! read_vector_from_xml<double>( descriptor, "vectorRPIDBN2", dbn2_data ) )
        {
          LOG_WARN( main_logger,"No rpiDBN2 data at " << descriptor->Row() << "\n");
        }
        else
        {
          kwxml.descriptor_rpidbn2() = dbn2_data;
        }
      }

      else if (string(descr_type) == "vectorKWSOTI3" )
      {
        vector< double > kwsoti3_data;
        if ( ! read_vector_from_xml<double>( descriptor, "vectorKWSOTI3", kwsoti3_data ) )
        {
          LOG_WARN( main_logger,"No KWSOTI3 data at " << descriptor->Row() << "\n");
        }
        else
        {
          kwxml.descriptor_kwsoti3() = kwsoti3_data;
        }
      }

      else if (string(descr_type) == "icsiHOG" )
      {
        vector< double > icsihog_data;
        if ( ! read_vector_from_xml<double>( descriptor, "vectorICSIHOG", icsihog_data ) )
        {
          LOG_WARN( main_logger,"No icsiHOG data at " << descriptor->Row() << "\n");
        }
        else
        {
          kwxml.descriptor_icsihog() = icsihog_data;
        }
      }

      else if (string(descr_type) == "texasHOG" )
      {
        vector< double > texashog_data;
        if ( ! read_vector_from_xml<double>( descriptor, "vectorTexasHOG", texashog_data ))
        {
          LOG_WARN( main_logger,"No texasHOG data at " << descriptor->Row() << "\n");
        }
        else
        {
          kwxml.descriptor_texashog() = texashog_data;
        }
      }

      else if (string(descr_type) == "cornell" )
      {
        vector< double > cornell_data;
        if ( ! read_vector_from_xml<double>( descriptor, "data", cornell_data ))
        {
          LOG_WARN( main_logger,"No cornell data at " << descriptor->Row() << "\n");
        }
        else
        {
          kwxml.descriptor_cornell() = cornell_data;
        }
      }

      else if (string(descr_type) == "pvoHOG" )
      {
        vector< double > pvo_hog_data;
        if ( ! read_vector_from_xml<double>( descriptor, "pvoRawScores", pvo_hog_data ))
        {
          LOG_WARN( main_logger,"No pvoHOG data at " << descriptor->Row() << "\n");
        }
        else
        {
          kwxml.descriptor_pvo_raw_scores() = pvo_hog_data;
        }
      }

      else if (string(descr_type) == "UMDLDS" )
      {
        vector< double > lds_data;
        if ( ! read_vector_from_xml<double>( descriptor, "data", lds_data ))
        {
          LOG_WARN( main_logger,"No UMDLDS data at " << descriptor->Row() << "\n");
        }
        else
        {
          kwxml.descriptor_umdlds() = lds_data;
        }
      }

      else if (string(descr_type) == "UMDPeriodicity" )
      {
        vector< double > per_data;
        if ( ! read_vector_from_xml<double>( descriptor, "data", per_data ))
        {
          LOG_WARN( main_logger,"No UMDPeriodicity data at " << descriptor->Row() << "\n");
        }
        else
        {
          kwxml.descriptor_umd_periodicity() = per_data;
        }
      }

      else if (string(descr_type) == "UMDSSM" )
      {
        vector< double > ssm_data;
        if ( ! read_vector_from_xml<double>( descriptor, "data", ssm_data ))
        {
          LOG_WARN( main_logger,"No UMDSSM data at " << descriptor->Row() << "\n");
        }
        else
        {
          kwxml.descriptor_umdssm() = ssm_data;
        }
      }

      else if (string(descr_type) == "raw" )
      {
        wmap.add_msg( "Ignoring raw descriptor until disambiguated between TRED and VIRAT" );
#if 0
        descriptor_raw_1d_type d;
        if ( ! read_vector_from_xml<double>(descriptor, "vector", d.data ))
        {
          LOG_WARN( main_logger, "No raw data at " << descriptor->Row() << "\n" );
        }
        else
        {
          kwxml.descriptor_raw_1d() = d;
        }
#endif
      }

      else if (string(descr_type) == "CUTIC" )
      {
        descriptor_cutic_type cutic;
        if ( ! read_vector_from_xml<double>( descriptor, "scoreClass", cutic.score_class ))
        {
          LOG_WARN( main_logger,"No scoreClass in CUTIC descriptor at " << descriptor->Row() << "\n");
        }
        else if ( ! read_vector_from_xml<int>( descriptor, "scoreType", cutic.score_type ))
        {
          LOG_WARN( main_logger,"No scoreType in CUTIC descriptor at " << descriptor->Row() << "\n");
        }
        else if ( ! read_vector_from_xml<double>( descriptor, "simTemporal", cutic.sim_temporal ))
        {
          LOG_WARN( main_logger,"No simTemporal in CUTIC descriptor at " << descriptor->Row() << "\n");
        }
        else if ( ! read_vector_from_xml<short>( descriptor, "descIndex", cutic.desc_index ))
        {
          LOG_WARN( main_logger,"No descIndex in CUTIC descriptor at " << descriptor->Row() << "\n");
        }
        else if ( ! read_vector_from_xml<double>( descriptor, "descRaw", cutic.desc_raw ))
        {
          LOG_WARN( main_logger,"No descRaw in CUTIC descriptor at " << descriptor->Row() << "\n");
        }
        else
        {
          kwxml.descriptor_cutic() = cutic;
        }
      }

      else if (string(descr_type) == "CUBoF" )
      {
        vector< double > bof_data;
        if ( ! read_vector_from_xml<double>( descriptor, "data", bof_data )) //said descRaw
        {
          LOG_WARN( main_logger,"No CUBoF data at " << descriptor->Row() << "\n");
        }
        else
        {
          kwxml.descriptor_cubof() = bof_data;
        }
      }

      else if (string(descr_type) == "CUTexture" )
      {
        vector< double > tex_data;
        if ( ! read_vector_from_xml<double>( descriptor, "data", tex_data )) //said descRaw
        {
          LOG_WARN( main_logger,"No CUTexture at " << descriptor->Row() << "\n");
        }
        else
        {
          kwxml.descriptor_cu_texture() = tex_data;
        }
      }

      else if (string(descr_type) == "CUColMoment" )
      {
        vector< double > colmom_data;
        if ( ! read_vector_from_xml<double>( descriptor, "data", colmom_data )) //said descRaw
        {
          LOG_WARN( main_logger,"No CUColMoment data at " << descriptor->Row() << "\n");
        }
        else
        {
          kwxml.descriptor_cu_col_moment() = colmom_data;
        }
      }

      else if (string(descr_type) == "overlap")
      {
        descriptor_overlap_type dot;
        bool good = true;
        TiXmlElement* e;
        e = descriptor->FirstChildElement( "src_trk_id" );
        if ( (!e) || (e->QueryUnsignedAttribute( "value", &dot.src_trk_id ) != TIXML_SUCCESS ))
        {
          LOG_WARN( main_logger, "No src_trk_id at " << descriptor->Row() << "\n" );
          good = false;
        }
        e = descriptor->FirstChildElement( "dst_trk_id" );
        if ( (!e) || (e->QueryUnsignedAttribute( "value", &dot.dst_trk_id ) != TIXML_SUCCESS ))
        {
          LOG_WARN( main_logger, "No dst_trk_id at " << descriptor->Row() << "\n" );
          good = false;
        }
        e = descriptor->FirstChildElement( "src_activity_id" );
        if ( (!e) || (e->QueryUnsignedAttribute( "value", &dot.src_activity_id ) != TIXML_SUCCESS ))
        {
          LOG_WARN( main_logger, "No src_activity_id at " << descriptor->Row() << "\n" );
          good = false;
        }
        e = descriptor->FirstChildElement( "dst_activity_id" );
        if ( (!e) || (e->QueryUnsignedAttribute( "value", &dot.dst_activity_id ) != TIXML_SUCCESS ))
        {
          LOG_WARN( main_logger, "No dst_activity_id at " << descriptor->Row() << "\n" );
          good = false;
        }
        e = descriptor->FirstChildElement( "n_frames_src" );
        if ( (!e) || (e->QueryUnsignedAttribute( "value", &dot.n_frames_src ) != TIXML_SUCCESS ))
        {
          LOG_WARN( main_logger, "No n_frames_src at " << descriptor->Row() << "\n" );
          good = false;
        }
        e = descriptor->FirstChildElement( "n_frames_dst" );
        if ( (!e) || (e->QueryUnsignedAttribute( "value", &dot.n_frames_dst ) != TIXML_SUCCESS ))
        {
          LOG_WARN( main_logger, "No n_frames_dst at " << descriptor->Row() << "\n" );
          good = false;
        }
        e = descriptor->FirstChildElement( "n_frames_overlap" );
        if ( (!e) || (e->QueryUnsignedAttribute( "value", &dot.n_frames_overlap ) != TIXML_SUCCESS ))
        {
          LOG_WARN( main_logger, "No n_frames_overlap at " << descriptor->Row() << "\n" );
          good = false;
        }
        e = descriptor->FirstChildElement( "mean_centroid_distance" );
        if ( (!e) || (e->QueryDoubleAttribute( "value", &dot.mean_centroid_distance ) != TIXML_SUCCESS ))
        {
          LOG_WARN( main_logger, "No mean_centroid_distance at " << descriptor->Row() << "\n" );
          good = false;
        }
        unsigned flag;
        e = descriptor->FirstChildElement( "radial_overlap_flag" );
        if ( (!e) || (e->QueryUnsignedAttribute( "value", &flag) != TIXML_SUCCESS ))
        {
          LOG_WARN( main_logger, "No n_frames_overlap at " << descriptor->Row() << "\n" );
          good = false;
        }
        else
        {
          dot.radial_overlap_flag = static_cast<bool>( flag );
        }
        e = descriptor->FirstChildElement( "mean_percentage_overlap" );
        if ( (!e) || (e->QueryDoubleAttribute( "value", &dot.mean_percentage_overlap ) != TIXML_SUCCESS ))
        {
          LOG_WARN( main_logger, "No mean_percentage_overlap at " << descriptor->Row() << "\n" );
          good = false;
        }

        // You know that sometime, somewhere, this will come to pass...
        // only check if good is still true (otherwise can't be sure dot.{src,dst}_activity_id is valid)
        if (good)
        {
          const map< size_t, string >& i2a = aries_interface::index_to_activity_map();
          typedef map< size_t, string >::const_iterator i2a_cit;
          string s_descriptor, s_system;

          e = descriptor->FirstChildElement( "src_activity_name" );
          if ( (!e) || (e->QueryStringAttribute( "value", &s_descriptor ) != TIXML_SUCCESS ))
          {
            LOG_WARN( main_logger, "No src_activity_name at " << descriptor->Row() << "\n" );
            good = false;
          }
          i2a_cit probe = i2a.find( dot.src_activity_id );
          s_system = (probe == i2a.end()) ? "" : probe->second;
          if ( s_descriptor != s_system )
          {
            LOG_WARN( main_logger, "Src activity index is " << dot.src_activity_id << ", named '" << s_descriptor
                      << "' in the file but is now '" << s_system << "' at " << descriptor->Row() );
            good = false;
          }
          e = descriptor->FirstChildElement( "dst_activity_name" );
          if ( (!e) || (e->QueryStringAttribute( "value", &s_descriptor ) != TIXML_SUCCESS ))
          {
            LOG_WARN( main_logger, "No dst_activity_name at " << descriptor->Row() << "\n" );
            good = false;
          }
          probe = i2a.find( dot.dst_activity_id );
          s_system = (probe == i2a.end()) ? "" : probe->second;
          if ( s_descriptor != s_system )
          {
            LOG_WARN( main_logger, "Dst activity index is " << dot.dst_activity_id << ", named '" << s_descriptor
                      << "' in the file but is now '" << s_system << "' at " << descriptor->Row() );
            good = false;
          }
        }

        if (good)
        {
          kwxml.descriptor_overlap() = dot;
        }
      }


      else if (string(descr_type) == "queryResultScore")
      {
        double d;
        TiXmlElement* e = descriptor->FirstChildElement( "score" );
        if ( (!e) || (e->QueryDoubleAttribute( "value", &d ) != TIXML_SUCCESS ))
        {
          LOG_WARN( main_logger,"No queryResultScore value at " << descriptor->Row() << "?\n");
        }
        else
        {
          kwxml.descriptor_query_result_score() = d;
        }
      }

      // Parse the metadata descriptor


      else if (string(descr_type) == "metadataDescriptor")
      {
        descriptor_metadata_type meta;
        TiXmlElement* e;

        e = descriptor->FirstChildElement( "GSD" );
        if ( !e )
        {
          meta.gsd = 0;
        } else if ( (e->QueryDoubleAttribute( "value", &meta.gsd ) != TIXML_SUCCESS ))
        {
          LOG_WARN( main_logger,"Could not read GSD at " << descriptor->Row() << "?\n");
        }
        e = descriptor->FirstChildElement( "SensorLatitude" );
        if ( (!e) || (e->QueryDoubleAttribute( "value", &meta.sensor_latitude ) != TIXML_SUCCESS ))
        {
          LOG_WARN( main_logger,"No SensorLatitude value at " << descriptor->Row() << "?\n");
        }
        e = descriptor->FirstChildElement( "SensorLongitude" );
        if ( (!e) || (e->QueryDoubleAttribute( "value", &meta.sensor_longitude ) != TIXML_SUCCESS ))
        {
          LOG_WARN( main_logger,"No SensorLongitude value at " << descriptor->Row() << "?\n");
        }
        e = descriptor->FirstChildElement( "UpperLeftCornerLatitude" );
        if ( (!e) || (e->QueryDoubleAttribute( "value", &meta.upper_left_corner_latitude ) != TIXML_SUCCESS ))
        {
          LOG_WARN( main_logger,"No UpperLeftCornerLatitude value at " << descriptor->Row() << "?\n");
        }
        e = descriptor->FirstChildElement( "UpperLeftCornerLongitude" );
        if ( (!e) || (e->QueryDoubleAttribute( "value", &meta.upper_left_corner_longitude ) != TIXML_SUCCESS ))
        {
          LOG_WARN( main_logger,"No UpperLeftCornerLongitude value at " << descriptor->Row() << "?\n");
        }
        e = descriptor->FirstChildElement( "UpperRightCornerLatitude" );
        if ( (!e) || (e->QueryDoubleAttribute( "value", &meta.upper_right_corner_latitude ) != TIXML_SUCCESS ))
        {
          LOG_WARN( main_logger,"No UpperRightCornerLatitude value at " << descriptor->Row() << "?\n");
        }
        e = descriptor->FirstChildElement( "UpperRightCornerLongitude" );
        if ( (!e) || (e->QueryDoubleAttribute( "value", &meta.upper_right_corner_longitude ) != TIXML_SUCCESS ))
        {
          LOG_WARN( main_logger,"No UpperRightCornerLongitude value at " << descriptor->Row() << "?\n");
        }
        e = descriptor->FirstChildElement( "LowerLeftCornerLatitude" );
        if ( (!e) || (e->QueryDoubleAttribute( "value", &meta.lower_left_corner_latitude ) != TIXML_SUCCESS ))
        {
          LOG_WARN( main_logger,"No LowerLeftCornerLatitude value at " << descriptor->Row() << "?\n");
        }
        e = descriptor->FirstChildElement( "LowerLeftCornerLongitude" );
        if ( (!e) || (e->QueryDoubleAttribute( "value", &meta.lower_left_corner_longitude ) != TIXML_SUCCESS ))
        {
          LOG_WARN( main_logger,"No LowerLeftCornerLongitude value at " << descriptor->Row() << "?\n");
        }
        e = descriptor->FirstChildElement( "LowerRightCornerLatitude" );
        if ( (!e) || (e->QueryDoubleAttribute( "value", &meta.lower_right_corner_latitude ) != TIXML_SUCCESS ))
        {
          LOG_WARN( main_logger,"No LowerRightCornerLatitude value at " << descriptor->Row() << "?\n");
        }
        e = descriptor->FirstChildElement( "LowerRightCornerLongitude" );
        if ( (!e) || (e->QueryDoubleAttribute( "value", &meta.lower_right_corner_longitude ) != TIXML_SUCCESS ))
        {
          LOG_WARN( main_logger,"No LowerRightCornerLongitude value at " << descriptor->Row() << "?\n");
        }
        e = descriptor->FirstChildElement( "HorizontalFieldOfView" );
        if ( (!e) || (e->QueryDoubleAttribute( "value", &meta.horizontal_field_of_view ) != TIXML_SUCCESS ))
        {
          LOG_WARN( main_logger,"No HorizontalFieldOfView value at " << descriptor->Row() << "?\n");
        }
        e = descriptor->FirstChildElement( "VerticalFieldOfView" );
        if ( (!e) || (e->QueryDoubleAttribute( "value", &meta.vertical_field_of_view ) != TIXML_SUCCESS ))
        {
          LOG_WARN( main_logger,"No VerticalFieldOfView value at " << descriptor->Row() << "?\n");
        }
        e = descriptor->FirstChildElement( "TimeStampMicrosecondsSince1970" );
        const char *str_time_stamp = 0;
        if ( (!e) || (str_time_stamp = e->Attribute( "value" )) == NULL )
        {
          LOG_WARN( main_logger,"No TimeStampMicrosecondsSince1970 value at " << descriptor->Row() << "?\n");
        }
        try
        {
          meta.timestamp_microseconds_since_1970 = std::stoll( str_time_stamp );
        }
        catch(...)
        {
          return false;
        }
        e = descriptor->FirstChildElement( "SlantRange" );
        if ( (!e) || (e->QueryFloatAttribute( "value", &meta.slant_range ) != TIXML_SUCCESS ))
        {
          LOG_WARN( main_logger,"No SlantRange value at " << descriptor->Row() << "?\n");
        }
        kwxml.descriptor_metadata() = meta;
      }


      else if (string(descr_type) == "motionDescriptor")
      {
        descriptor_motion_type d;
        double *ptrs[] = { &d.ground_pos_x,
                           &d.ground_pos_y,
                           &d.ground_speed,
                           &d.ground_acceleration,
                           &d.heading,
                           &d.delta_heading,
                           &d.exp_heading,
                           &d.ang_momentum,
                           &d.curvature,
                           0 };
        const char *tags[] = { "GroundSmoothPosX",
                               "GroundSmoothPosY",
                               "GroundSpeed",
                               "GroundAcceleration",
                               "Heading",
                               "DeltaHeading",
                               "ExpHeading",
                               "AngMomentum",
                               "Curvature",
                               0 };

        for (unsigned d_index = 0; ptrs[d_index] != 0; ++d_index)
        {
          TiXmlElement* e = descriptor->FirstChildElement( tags[d_index] );
          if ( (!e) || (e->QueryDoubleAttribute( "value", ptrs[d_index] ) != TIXML_SUCCESS))
          {
            LOG_WARN( main_logger,"No " << tags[d_index] << " at " << descriptor->Row() << "in motionDescriptor\n");
          }
        }
        kwxml.descriptor_motion() = d;
      }

      else
      {
        if( descr_type )
        {
          wmap.add_msg( string("Unknown descriptor type '") + descr_type + string("'") );
        }
        else
        {
          LOG_WARN( main_logger,"At " << descriptor->Row()
                   << ": descriptor has no type\n");
        }
      }
    } // ...for each descriptor

  } // for each track object;

  wmap.dump_msgs();


  // all done!
  return true;

}

} // ...track_oracle
} // ...kwiver
