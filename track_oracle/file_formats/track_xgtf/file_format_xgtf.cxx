/*ckwg +5
 * Copyright 2012-2014 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "file_format_xgtf.h"

#include <vgl/vgl_point_2d.h>

#include <tinyxml.h>

#include <track_oracle/utils/tokenizers.h>
#include <track_oracle/utils/logging_map.h>
#include <track_oracle/file_formats/track_xgtf/track_xgtf.h>
#include <track_oracle/aries_interface/aries_interface.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <utility>

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

using std::multimap;
using std::pair;
using std::map;
using std::string;
using std::make_pair;
using std::istringstream;
using std::vector;

namespace // start of anonymous namespace
{

enum xgtf_style { ACTIVITY, APPEARANCE };

typedef unsigned int activity_type;

//
// redefinition of tinyxml elements for style consistency
//
typedef TiXmlNode xml_node_t;
typedef TiXmlElement xml_element_t;
typedef TiXmlDocument xml_document_t;
typedef TiXmlHandle xml_handle_t;


//
// local structs for viper's activity maps
//

typedef multimap< pair<unsigned int, unsigned int>, activity_type > viper_activity_list_t;
typedef multimap< pair<unsigned int, unsigned int>, activity_type >::iterator viper_activity_list_it;
typedef multimap< pair<unsigned int, unsigned int>, activity_type >::const_iterator viper_activity_list_c_it;

//
// pair up the nodes with their "name" attributes -- probably an easier way
// to do this in tinyXML?
//

typedef map< string, xml_element_t* > named_nodes_t;
typedef map< string, xml_element_t* >::const_iterator named_nodes_c_it;

//
// build up a map of the immediate children of this node,
// cast to elements, and uniquely identified by name
//

bool
build_named_node_map( xml_node_t* node,
                      named_nodes_t& named_nodes )
{
    xml_node_t* xml_child_node = 0;
    while ( ( xml_child_node = node->IterateChildren( xml_child_node ) ) )
    {
      // it's an element?
      xml_element_t* e = xml_child_node->ToElement();
      if ( !e )
      {
        LOG_ERROR( main_logger, "Error reading line " << xml_child_node->Row()
                   << " could not be converted into an element" );
        return false;
      }

      // it has a name attribute?
      const char* attr_name = e->Attribute( "name" );
      if ( !attr_name )
      {
        LOG_ERROR( main_logger, "Error reading line " << xml_child_node->Row()
                   << " has no 'name' attribute" );
        return false;
      }

      // it's a unique name?
      named_nodes_c_it probe = named_nodes.find( attr_name );
      if ( probe != named_nodes.end() )
      {
        LOG_ERROR( main_logger, "Error reading line " << xml_child_node->Row()
                   << " has duplicate '" << attr_name << "' attributes" );
        return false;
      }

      // remember it!
      named_nodes.insert( make_pair( attr_name, e ) );
    }
    return true;
}

//
// Given a framespan "m:n", e.g. "102:694", parse out the start and end frames
//

bool
parse_framespan( const string& framespan_str,
                 pair<unsigned int, unsigned int>& span )
{
  istringstream iss( framespan_str );
  char colon, junk;
  bool okay = false;

  if ( iss >> span.first >> colon >> span.second )
  {
    if ( !(iss >> junk) != 0 )
    {
      okay = true;
    }
  }
  if ( !okay )
  {
    LOG_ERROR( main_logger, "Couldn't parse framespan '" << framespan_str << "'" );
  }

  return okay;
}

//
// Given a framespan "m:n", e.g. "102:694", parse out the start and end frames
// This time, allow for multiple sets: "a:b c:d ..."
//

bool
parse_framespan_set( const string& framespan_str,
                     vector< pair<unsigned int, unsigned int> > & span_set )
{
  istringstream iss( framespan_str );

  char colon;
  bool okay = false;
  pair< unsigned int, unsigned int> span;

  while ( iss >> span.first >> colon >> span.second )
  {
    span_set.push_back( span );
    okay = true;
  }
  if ( !okay )
  {
    LOG_ERROR( main_logger, "Couldn't parse framespan '" << framespan_str << "'" );
  }

  return okay;
}

//
// Given a bounding box node, e.g.
// <data:bbox framespan="2015:2015" height="22" width="20" x="322" y="263"/>
// ... extract the frame span and the bounding box.
//
bool
extract_bounding_box( xml_node_t* xml_node,
                      pair< unsigned int, unsigned int >& span,
                      vgl_box_2d< double >& box )
{
  xml_element_t* e = xml_node->ToElement();
  if ( !e )
  {
    LOG_ERROR( main_logger, "extract_bounding_box: Couldn't convert xmlnode to element?" );
    return false;
  }
  if ( !parse_framespan( e->Attribute( "framespan" ), span ) )
  {
    LOG_ERROR( main_logger, "extract_bounding_box: Couldn't extract framespan?" );
    return false;
  }
  int h, w, x, y;
  if ( e->QueryIntAttribute( "height", &h ) != TIXML_SUCCESS )
  {
    LOG_ERROR( main_logger, "extract_bounding_box: Couldn't extract height?" );
    return false;
  }
  if ( e->QueryIntAttribute( "width", &w ) != TIXML_SUCCESS )
  {
    LOG_ERROR( main_logger, "extract_bounding_box: Couldn't extract width?" );
    return false;
  }
  if ( e->QueryIntAttribute( "x", &x ) != TIXML_SUCCESS )
  {
    LOG_ERROR( main_logger, "extract_bounding_box: Couldn't extract x?" );
    return false;
  }
  if ( e->QueryIntAttribute( "y", &y ) != TIXML_SUCCESS )
  {
    LOG_ERROR( main_logger, "extract_bounding_box: Couldn't extract y?" );
    return false;
  }

  box.set_min_point( vgl_point_2d< double >( x, y ) );
  box.set_max_point( vgl_point_2d< double >( x+w, y+h ) );

  return true;
}

//
// given an occlusion node, e.g.
// <data:fvalue framespan="61:277" value="1.0"/>
// ...extract the frame span and the occlusion value.
//
bool
extract_occlusion( xml_node_t* xml_node,
                   pair< unsigned int, unsigned int >& span,
                   double& occlusion_value )
{
  xml_element_t* e = xml_node->ToElement();
  if ( !e )
  {
    LOG_ERROR( main_logger, "extract_occlusion: Couldn't convert xmlnode to element?" );
    return false;
  }
  if ( !parse_framespan( e->Attribute("framespan"), span ) )
  {
    LOG_ERROR( main_logger, "extract_occlusion: Couldn't extract framespan?" );
    return false;
  }
  if ( e->QueryDoubleAttribute( "value", &occlusion_value ) != TIXML_SUCCESS )
  {
    LOG_ERROR( main_logger, "extract_occlusion: Couldn't extract value?" );
    return false;
  }
  return true;
}


//
// named_nodes is a map of the XML nodes which are children of
// this_track.  Extract the "Location" nodes and "Occlusion"
// nodes, and associate the frame ID, bounding box, and optional
// occlusion values with the frames.
//

bool
extract_viper_frame_data( const named_nodes_t& named_nodes,
                          ::kwiver::track_oracle::track_handle_type& this_track,
                          map< unsigned int, ::kwiver::track_oracle::frame_handle_type>& xgtf_frame_map )
{
  static ::kwiver::track_oracle::track_xgtf_type xgtf_schema;
  xgtf_frame_map.clear();

  // first, pull the bounding boxes from "Location"
  named_nodes_c_it probe = named_nodes.find( "Location" );
  if ( probe == named_nodes.end() )
  {
    LOG_ERROR( main_logger, "Couldn't find Location node?" );
    return false;
  }

  vgl_box_2d< double > box;
  pair< unsigned int, unsigned int > span;

  xml_node_t* xml_location_node = probe->second;
  xml_node_t* xml_bbox_node = 0;

  while ( ( xml_bbox_node = xml_location_node->IterateChildren( xml_bbox_node ) ) )
  {
    // extract the bounding box from the XML
    if ( !extract_bounding_box( xml_bbox_node, span, box ) )
    {
      return false;
    }

    // create a box for each frame in the span
    for ( unsigned frame = span.first; frame <= span.second; ++frame )
    {
      ::kwiver::track_oracle::frame_handle_type frame_handle = xgtf_schema( this_track ).create_frame();
      xgtf_schema[ frame_handle ].bounding_box() = box;
      xgtf_schema[ frame_handle ].frame_number() = frame;
      xgtf_frame_map[ frame ] = frame_handle;
    }
  }

  // If "Occlusion" is present, read and associate with frames
  probe = named_nodes.find( "Occlusion" );
  if ( probe != named_nodes.end() )
  {
    xml_node_t* xml_occlusion_node = probe->second;
    xml_node_t* xml_fval_node = 0;
    double occ_value;

    while ( ( xml_fval_node = xml_occlusion_node->IterateChildren( xml_fval_node ) ) )
    {
      // extract span and occlusion
      if ( !extract_occlusion( xml_fval_node, span, occ_value ))
      {
        return false;
      }

      // associate it with the frames.  Only set occlusions where there are
      // bounding boxes (assuming the MITRE annotators were... non-specific.)
      for ( unsigned int frame = span.first; frame <= span.second; ++frame )
      {
        map< unsigned, ::kwiver::track_oracle::frame_handle_type >::const_iterator frame_it = xgtf_frame_map.find( frame );
        if ( frame_it != xgtf_frame_map.end() )
        {
          xgtf_schema[ frame_it->second ].occlusion() = occ_value;
        }
      }
    }
  }

  return true;
}

//
// Assume that all children of the sourcefile which are not "Location" or
// "Occlusion" or any of the other Not-An-Activity-Tags are activities, e.g.
//
//  <attribute name="Standing">
//    <data:bvalue framespan="61:1715" value="true"/>
//    <data:bvalue framespan="1716:2016" value="false"/>
//  </attribute>
//
// ... Associate the "true" framespans with their activities.  No frame should
// have more than one activity.  If a frame has no activity... warn?
// CHECK for more than one activity!
// clip to object span frame values
//

bool
extract_viper_activities( const named_nodes_t& named_nodes,
                          viper_activity_list_t& m,
                          const pair< unsigned int, unsigned int>& object_span,
                          unsigned int /*viperID*/,
                          bool promote_pvmoving,
                          xgtf_style style,
                          ::kwiver::logging_map_type& warnings )
{
  // get the current string-to-index map for VIRAT
  size_t PERSON_MOVING_INDEX = ::kwiver::track_oracle::aries_interface::activity_to_index( "PersonMoving" );
  size_t VEHICLE_MOVING_INDEX = ::kwiver::track_oracle::aries_interface::activity_to_index( "VehicleMoving" );

  // start walking down the nodes
  for ( named_nodes_c_it n = named_nodes.begin(); n != named_nodes.end(); ++n )
  {
    if ( n->first == "Location" ) continue;
    if ( n->first == "Occlusion" ) continue;
    if ( n->first == "Event-Related Occlusion" ) continue;
    if ( n->first == "FOV" ) continue;
    if ( n->first == "Wavelength" ) continue;

    // assume it's an activity
    string activity_name = n->first;
    xml_node_t* xml_bvalue_node = 0;
    while ( ( xml_bvalue_node = n->second->IterateChildren( xml_bvalue_node ) ) )
    {
      xml_element_t* e = xml_bvalue_node->ToElement();
      if ( !e )
      {
        LOG_ERROR( main_logger, "extract_viper_activities: couldn't convert " << activity_name
                   << " bvalues to element?" );
        return false;
      }
      // can we convert it to an activity name?

      // check for appearance-based annotations
      if ( style == APPEARANCE )
      {
        if ( activity_name.find( "suv" ) != string::npos ||
             activity_name.find( "pickup" ) != string::npos ||
             activity_name.find( "car" ) != string::npos ||
             activity_name.find( "vehicle" ) != string::npos )
        {
          activity_name = "VEHICLE_MOVING";
        }
        else if ( activity_name.find( "person" ) != string::npos )
        {
          activity_name = "PERSON_MOVING";
        }
      }
      else
      {
        // special cases for the, um, idiosyncrasies of other performers
        // in choosing labels for annotations
        if ( activity_name == "Entering a Vehicle" ) activity_name = "Getting Into a Vehicle";
        else if ( activity_name == "Exiting a Vehicle" ) activity_name = "Getting Out of a Vehicle";
        else if ( activity_name == "Person Entering a Facility" ) activity_name = "Entering a Facility";
        else if ( activity_name == "Person Exiting a Facility" ) activity_name = "Exiting a Facility";
        else if ( activity_name == "Environment-Related Occlusion" ) activity_name = "Not Scored";

        // adjust for ARIES aliasing (sigh)
        if ( activity_name == "Pulling" ) activity_name = "PERSON_PULLING";
        else if ( activity_name == "Carrying Together" ) activity_name = "PERSON_CARRYING_TOGETHER";
        else if ( activity_name == "Climbing Atop" ) activity_name = "PERSON_CLIMBING_ATOP";
        else if ( activity_name == "Driving into a Facility" ) activity_name = "VEHICLE_DRIVING_INTO_A_FACILITY";
        else if ( activity_name == "Driving out of a Facility" ) activity_name = "VEHICLE_DRIVING_OUT_OF_A_FACILITY";
        else if ( activity_name == "Kicking" ) activity_name = "PERSON_KICKING";
        else if ( activity_name == "Laying Wire" ) activity_name = "PERSON_LAYING_WIRE";
        else if ( activity_name == "Looping" ) activity_name = "VEHICLE_LOOPING";
        else if ( activity_name == "Maintaining Distance" ) activity_name = "VEHICLE_MAINTAINING_DISTANCE";
        else if ( activity_name == "Passing" ) activity_name = "VEHICLE_PASSING";
        else if ( activity_name == "Pushing" ) activity_name = "PERSON_PUSHING";
        else if ( activity_name == "Sitting" ) activity_name = "PERSON_SITTING";
        else if ( activity_name == "Throwing" ) activity_name = "PERSON_THROWING";
      }

      bool skip_flag = false;
      if ( activity_name == "No. Vehicle" )
      {
        skip_flag = true;
      }

      // For LMCO's VIRAT phase 3 "golden ground truth", there are tags such
      // as "Running No Score".  We'll skip those AS LONG as the value is false.
      {
        string no_score_str( " No Score" );
        size_t len_an( activity_name.size() ), len_ns( no_score_str.size() );
        const string true_str( "true" );
        bool ends_in_no_score =
          ( len_an >= len_ns ) &&
          ( activity_name.substr( len_an-len_ns, len_ns ) == no_score_str );
        bool is_true = ( e->Attribute( "value" ) == true_str );

        if ( ends_in_no_score && ( ! is_true ) )
        {
          // silently skip
          skip_flag = true;
        }
      }

      if ( !skip_flag )
      {
        pair<bool, unsigned int> act_type( false, 0 );
        pair<bool, unsigned int> promote_type( false, 0 );

        try
        {
          act_type.second = static_cast<unsigned int>(
            ::kwiver::track_oracle::aries_interface::activity_to_index( activity_name ) );
          // if we get here, it worked
          act_type.first = true;

          if ( ::kwiver::track_oracle::aries_interface::promote_to_PERSON_MOVING( act_type.second ) )
          {
            promote_type.first = true;
            promote_type.second = PERSON_MOVING_INDEX;
          }
          else if ( ::kwiver::track_oracle::aries_interface::promote_to_VEHICLE_MOVING( act_type.second ) )
          {
            promote_type.first = true;
            promote_type.second = VEHICLE_MOVING_INDEX;
          }
          else
          {
            warnings.add_msg( activity_name + " not promoted" );
          }
        }
        catch ( ::kwiver::track_oracle::aries_interface_exception& aries_type_exception )
        {
          warnings.add_msg( string( "unrecognized activity: " ) + activity_name + ": " + aries_type_exception.what() );
          if ( activity_name == "Bicycling" )
          {
            promote_type.first = true;
            promote_type.second = PERSON_MOVING_INDEX;
          }
        }

        //
        // new promotion logic:
        // -- only promote if promote_pvmoving is set
        if ( promote_pvmoving )
        {
          // ...in which case, we REPLACE the activity we're reading
          // with PVMoving so as ensure that an xgtf with N tracks
          // going in has N tracks going out, rather than 2N tracks
          // (the "real" track and its PVMoving "shadow") as previously done.
          if ( promote_type.first )
          {
            act_type = promote_type;
          }
        }

        // do we have to do anything?
        if ( !act_type.first )
        {
          // no... nothing to do
          continue;
        }

        // only take the "true" values
        const string true_str( "true" );
        if ( e->Attribute( "value" ) == true_str )
        {
          pair< unsigned int, unsigned int > span;
          if ( !parse_framespan( e->Attribute("framespan"), span ) )
          {
            LOG_ERROR( main_logger, "Couldn't parse framespan for " << activity_name << "?" );
            return false;
          }
          if ( span.first < object_span.first )
          {
            span.first = object_span.first;
          }
          if ( object_span.second < span.second )
          {
            span.second = object_span.second;
          }

          // if defined, add an entry for the base activity
          if ( act_type.first )
          {
            m.insert( make_pair( span, act_type.second ) );
          }

        } // ...if true
      } // ... if not skipped
    } // ...for all value nodes
  } // ...for all named nodes

  return true;
}

xml_node_t*
doc_to_source_node( const string& filename, xml_document_t& doc )
{
  xml_node_t* xml_root = doc.RootElement();
  if ( !xml_root )
  {
    LOG_ERROR( main_logger, "Couldn't load root element from '" << filename << "'; skippig" );
    return 0;
  }

  xml_node_t* xml_data = xml_root->FirstChild( "data" );
  if ( !xml_data )
  {
    LOG_ERROR( main_logger, "Couldn't find the 'data' child in '" << filename << "'; skipping" );
    return 0;
  }

  // ...down to the sourcefile node.  For now, assume a single sourcefile
  xml_node_t* xml_source = xml_data->FirstChild( "sourcefile" );
  if ( !xml_source )
  {
    LOG_ERROR( main_logger, "Couldn't find the 'sourcefile' child in '" << filename << "'; skipping" );
    return 0;
  }

  return xml_source;
}


} // anon


namespace kwiver {
namespace track_oracle {

xgtf_reader_opts&
xgtf_reader_opts
::operator=( const file_format_reader_opts_base& rhs_base )
{
  const xgtf_reader_opts* rhs = dynamic_cast<const xgtf_reader_opts*>(&rhs_base);

  if (rhs)
  {
    this->set_promote_pvmoving( rhs->promote_pvmoving );
  }
  else
  {
    LOG_WARN(main_logger, "Assigned a non-xgtf options structure to a xgtf options structure: Slicing the class");
  }

  return *this;
}

bool
file_format_xgtf
::inspect_file( const string& fn ) const
{
  vector< string > tokens = xml_tokenizer::first_n_tokens( fn, 10 );

  for ( size_t i=0; i<tokens.size(); ++i )
  {
    if ( tokens[i].find( "<viper" ) != string::npos )
    {
      return true;
    }
  }

  return false;
}

bool
file_format_xgtf
::read( const string& fn,
        track_handle_list_type& tracks ) const
{
  logging_map_type warnings( main_logger, KWIVER_LOGGER_SITE );
  track_xgtf_type xgtf_schema;

  tracks.clear();

  // dig through the XML wrappers...
  xml_document_t doc( fn.c_str() );
  xml_handle_t doc_handle( &doc );

  if ( !doc.LoadFile() )
  {
    LOG_ERROR( main_logger, "TinyXML couldn't load '" << fn << "'; skipping" );
    LOG_ERROR( main_logger, "Error description: " << doc.ErrorDesc() );
    LOG_ERROR( main_logger, "Error location: row " << doc.ErrorRow() << "; col " << doc.ErrorCol() );
    return false;
  }

  xml_node_t* xml_source = doc_to_source_node( fn, doc );

  if ( !xml_source )
  {
    return false;
  }

  // does this xml file contain activity or appearance annotations?
  xgtf_style style = ACTIVITY;

  xml_node_t* xml_viper_object = 0;
  while ( ( xml_viper_object = xml_source->IterateChildren( xml_viper_object ) ) )
  {
    // only interested in Object nodes
    const string object_str( "object" );
    if ( xml_viper_object->Value() != object_str ) continue;

    //
    // Consider the VIPER format.  Each child of a sourcefile has:
    // - attribute ID
    // - attribute 'name' (OBJECT or PERSON)
    // - attribute framespan, [objM:objN]
    // - a set of children tagged 'attribute':
    // --- one named 'Location'
    // ------ whose children are 'data:bbox' with framespan and bounding box as the attributes
    // --- one named 'Occlusion'
    // ------ whose children are 'data:fvalue' with framespan and occluded value as the attributes
    // --- one for each of the 20 activities
    // ------ whose children are 'data:bvalue' with framespan [actM:actN] and true/false as the attributes
    //
    // At any given frame, exactly one of the bvalues should be true.
    //
    // We'll split out tracks based on the framespan of the bvalues,
    // keeping in mind that each bvalue should be clipped by the object framespan.
    //

    // =====
    // 1) break out the invariant id and 'class' (object/person).
    // =====

    xml_element_t* xmle = xml_viper_object->ToElement();
    if ( !xmle ) return false;

    int viperID = -1;
    if ( xmle->QueryIntAttribute( "id", &viperID ) != TIXML_SUCCESS ) return false;

    string viper_class( xmle->Attribute( "name" ));
    if ( viper_class.empty() ) return false;

    // special-case for "augmented" performer annotations
    if ( viper_class == "PERSON-VEHICLE" ) viper_class = "PERSON";
    if ( viper_class == "PERSON-FACILITY" ) viper_class = "PERSON";

    // determine if gt is in appearance form divided into static and moving categories
    if ( viper_class == "MOVER" ||
         viper_class == "STATIC" )
    {
      style = APPEARANCE;
    }
    // it contains activities so it should now be either "PERSON" or "VEHICLE"
    // ...or, now with AFRL's ESC evaluation XGTF, No_Annotation_Zone or
    // Environment_Induced_Movement
    else if ( ( viper_class != "PERSON" ) &&
              ( viper_class != "VEHICLE" ) &&
              ( viper_class != "No_Annotation_Zone" ) &&
              ( viper_class != "Environment_Induced_Movement" ) )
    {
      LOG_ERROR( main_logger, "XGTF reader: file '" << fn << "' row " << xmle->Row()
                 << " track " << viperID << ":unknown class '" << viper_class << "'" );
      return false;
    }

    // =====
    // 2) need to pull the object framespan for clipping
    // =====
    //

    // framespan is REQUIRED
    if ( !xmle->Attribute( "framespan" ) )
    {
      LOG_WARN( main_logger, "xgtf_reader: viper_class " << viper_class << " has no framespan near row "
                << xmle->Row() << "?  Skipping" );
      continue;
    }

    pair<unsigned int, unsigned int> object_span;

    // annoyingly, framespans may be multiple sets: "a:b c:d e:f ..."
    // coalesce into the first and last elements
    vector< pair<unsigned int, unsigned int> > objectspan_set;
    if ( !parse_framespan_set( xmle->Attribute( "framespan" ), objectspan_set ) )
    {
      LOG_ERROR( main_logger, "...while reading '" << fn << "'" );
      return false;
    }
    object_span.first = objectspan_set.front().first;
    object_span.second = objectspan_set.back().second;

    // =====
    // 3) Gather all the children nodes up: one "location", one "occlusion"
    // all the activities
    // =====

    named_nodes_t named_nodes;
    if ( !build_named_node_map( xml_viper_object, named_nodes ) )
    {
      LOG_ERROR( main_logger, "...while reading '" << fn << "'" );
      return false;
    }

    // =====
    // 4) From the children nodes of this (track) node, generate the list of frames
    // =====

    // read in the (possibly multi-activity) xgtf_track
    track_handle_type xgtf_track = xgtf_schema.create();
    map< unsigned, frame_handle_type > xgtf_frame_map;
    if ( !extract_viper_frame_data( named_nodes, xgtf_track, xgtf_frame_map ) )
    {
      LOG_ERROR( main_logger, "...while reading '" << fn << "'" );
      return false;
    }

    // =====
    // 5) Split into per-activity framespans, clipped by the object framespan
    // =====

    viper_activity_list_t activities;
    if ( !extract_viper_activities( named_nodes,
                                    activities,
                                    object_span,
                                    viperID,
                                    this->opts.promote_pvmoving,
                                    style,
                                    warnings ) )
    {
      LOG_ERROR( main_logger, "...while reading '" << fn << "'" );
      return false;
    }

    // =====
    // 6) For each activity, create a track
    // =====


    ::kwiver::track_oracle::track_xgtf_type dst_schema;
    for ( viper_activity_list_c_it act_it = activities.begin(); act_it != activities.end(); ++act_it )
    {
      //
      // The VIPER GUI seems prone to accidentally creating single-frame tracks.
      // Skip them here.
      //
      if ( act_it->first.first == act_it->first.second )
      {
        continue;
      }

      track_handle_type this_track = xgtf_schema.create();
      xgtf_schema( this_track ).activity() = act_it->second;
      xgtf_schema( this_track ).activity_probability() = 1.0;
      xgtf_schema( this_track ).frame_span() = act_it->first;

      // No enum yet for PVO, store the string

      xgtf_schema( this_track ).type() = viper_class;
      xgtf_schema( this_track ).external_id() = viperID;

      // copy over the frame data
      for ( unsigned int frame = act_it->first.first; frame <= act_it->first.second; ++frame )
      {
        // look up the frame in the overall xgtf track
        map< unsigned, frame_handle_type >::const_iterator probe = xgtf_frame_map.find( frame );
        if ( probe == xgtf_frame_map.end() )
        {
          const string msg = "frame with no data";
          bool emit_warning = warnings.add_msg( msg );
          if ( emit_warning )
          {
            LOG_WARN( main_logger, "xgtf_reader: "
                      << "id: " << viperID << " event: " << act_it->second
                      << " frame: " << frame  << " (framespan " << act_it->first.first
                      << ":" << act_it->first.second << ") has no data? "
                      << "(warning only printed once) Skipping..." );
          }
          continue;
        }

        const frame_handle_type& lookup_frame = probe->second;

        frame_handle_type this_frame = xgtf_schema( this_track ).create_frame();

        dst_schema[ this_frame ].bounding_box() = xgtf_schema[ lookup_frame ].bounding_box();
        dst_schema[ this_frame ].frame_number() = xgtf_schema[ lookup_frame ].frame_number();
        if ( dst_schema[ lookup_frame ].occlusion.exists() )
        {
          dst_schema[ this_frame ].occlusion() = xgtf_schema[ lookup_frame ].occlusion();
        }
      }

      // add the track to the return set
      tracks.push_back( this_track );
    }

    // 7) Delete the original track we saved in track_oracle...
    // double-check    xgtf(this_track).remove_me();

  } // ...for each sourcefile

  if ( !warnings.empty() )
  {
    LOG_INFO( main_logger, "xgtf_reader: Warnings from loading '"  << fn << "':");
    warnings.dump_msgs();
    LOG_INFO( main_logger, "xgtf_reader: end of warnings" );
  }

  // all done!
  return true;

}

} // ...track_oracle
} // ...kwiver
