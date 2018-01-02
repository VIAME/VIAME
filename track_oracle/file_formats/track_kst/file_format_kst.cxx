/*ckwg +5
 * Copyright 2012-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "file_format_kst.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <ctype.h>

#include <vgl/vgl_box_2d.h>
#include <vgl/vgl_point_2d.h>
#include <vul/vul_timer.h>

#include <vital/util/string.h>

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

using std::getline;
using std::ifstream;
using std::istream;
using std::istringstream;
using std::make_pair;
using std::pair;
using std::string;
using std::vector;

namespace { // anon

bool
get_next_nonblank_line( istream& is, string& line )
{
  while ( getline(is, line) )
  {
    kwiver::vital::left_trim(line);
    // skip blank lines
    if (line.empty())
    {
      continue;
    }
    // skip comments
    if (line[0] == '#')
    {
      continue;
    }

    return true;
  }
  return false;
}


// each node in the KST parse tree is a vector of elements;
// each element is either a string or a pointer to another node.

enum kst_etype { KST_STRING, KST_NODE_PTR };

struct kst_element
{
  virtual kst_etype get_etype() const = 0;
  virtual ~kst_element() {}
};

struct kst_node
{
  vector< kst_element* > elist;
  ~kst_node()
  {
    for (size_t i=0; i<elist.size(); ++i)
    {
      delete elist[i];
    }
  }

  bool e_is_str( unsigned index, string& s );
  bool e_is_int( unsigned index, int& i );
  bool e_is_double( unsigned index, double& d );
  bool e_is_long_long( unsigned index, unsigned long long& ts );
  bool e_is_node( unsigned index, kst_node*& ptr );
  bool e_is_vector( unsigned index, vector< double >& d );
  bool e_is_box( unsigned index, vgl_box_2d<double>& box );

private:
  kst_element* get_element( unsigned index );
  bool get_string( unsigned index, string& s );

};

struct kst_string: public kst_element
{
private:
  string s;
public:
  kst_string( const string& str ): s(str) {}
  kst_etype get_etype() const { return KST_STRING; }
  string get_str() const { return s; }
};

struct kst_node_ptr: public kst_element
{
private:
  kst_node* n;
public:
  kst_node_ptr( kst_node* node ): n(node) {}
  ~kst_node_ptr() { delete n; }
  kst_etype get_etype() const { return KST_NODE_PTR; }
  kst_node* get_node_ptr() const { return n; }
};

kst_element*
kst_node
::get_element( unsigned index )
{
  if ( index >= this->elist.size() )
  {
    LOG_ERROR( main_logger, "Attempt to access node " << index << " in element list of size " << this->elist.size() );
    return 0;
  }
  return this->elist[ index ];
}

bool
kst_node
::get_string( unsigned index, string& s )
{
  kst_element* e = this->get_element( index );
  if ( ! e ) return false;
  if ( e->get_etype() != KST_STRING )
  {
    LOG_ERROR( main_logger, "Tried to access node as " << KST_STRING << "; found " << e->get_etype() );
    return false;
  }
  kst_string* es = dynamic_cast< kst_string *>( e );
  s = es->get_str();
  return true;
}

bool
kst_node
::e_is_str( unsigned index, string& s )
{
  return this->get_string( index, s );
}

bool
kst_node
::e_is_int( unsigned index, int& i )
{
  string es;
  if ( ! this->get_string( index, es )) return false;
  istringstream iss( es );
  if ( ! ( iss >> i ))
  {
    LOG_ERROR( main_logger, "Couldn't parse int from '" << es << "'" );
    return false;
  }
  return true;
}

bool
kst_node
::e_is_double( unsigned index, double& d )
{
  string es;
  if ( ! this->get_string( index, es )) return false;
  istringstream iss( es );
  if ( ! ( iss >> d ))
  {
    LOG_ERROR( main_logger, "Couldn't parse double from '" << es << "'" );
    return false;
  }
  return true;
}

bool
kst_node
::e_is_long_long( unsigned index, unsigned long long& ts )
{
  string es;
  if ( ! this->get_string( index, es )) return false;
  istringstream iss( es );
  if ( ! ( iss >> ts ))
  {
    LOG_ERROR( main_logger, "Couldn't parse unsigned long long from '" << es << "'" );
    return false;
  }
  return true;
}

bool
kst_node
::e_is_node( unsigned index, kst_node*& ptr )
{
  kst_element* e = this->get_element( index );
  if ( ! e ) return false;
  if ( e->get_etype() != KST_NODE_PTR )
  {
    LOG_ERROR( main_logger, "Tried to access node as " << KST_NODE_PTR << "; found " << e->get_etype() );
    return false;
  }
  kst_node_ptr* n = dynamic_cast< kst_node_ptr *>( e );
  ptr = n->get_node_ptr();
  return true;
}

bool
kst_node
::e_is_vector( unsigned index, vector< double >& d )
{
  kst_node* n;
  if ( ! this->e_is_node( index, n )) return false;
  d.clear();
  for (unsigned i=0; i<n->elist.size(); ++i)
  {
    double tmp;
    if ( ! n->e_is_double( i, tmp )) return false;
    d.push_back( tmp );
  }
  return true;
}

bool
kst_node
::e_is_box( unsigned index, vgl_box_2d<double>& box )
{
  kst_node *n;
  if ( ! this->e_is_node( index, n )) return false;
  if ( n->elist.size() != 2 )
  {
    LOG_ERROR( main_logger, "Parsing box; expected two elements; found " << n->elist.size() );
    return false;
  }
  vector<double> ul, lr;
  if ( ! n->e_is_vector( 0, ul )) return false;
  if ( ! n->e_is_vector( 1, lr )) return false;

  if ( ul.size() != 2)
  {
    LOG_ERROR( main_logger, "Parsing box; upper left contains " << ul.size() << " elements; expected 2" );
    return false;
  }
  if ( lr.size() != 2)
  {
    LOG_ERROR( main_logger, "Parsing box; lower right contains " << lr.size() << " elements; expected 2" );
    return false;
  }

  box = vgl_box_2d<double>(
    vgl_point_2d<double>( ul[1], ul[0] ),
    vgl_point_2d<double>( lr[1], lr[0] ) );

  return true;
}

pair< string, bool >
get_token( istream& is )
{
  // tokens are:
  // 1) unquoted '['
  // 2) unquoted ']'
  // otherwise, chomp whitespace until:
  // 3) any non-space character sequence delimted by a comma or a semicolon
  //
  // Complications:
  // '#' to end-of-line is a comment and ignored
  // output of case (3) above have any opening/closing double-quotes removed.
  // return pair (token, status), where status is false if !is.good

  const unsigned token_buf_size = 256;
  char token_buf[token_buf_size];
  unsigned int token_index=0;
  bool chomping_whitespace = true;
  bool in_quotes = false;
  char c;
  while (is.good())
  {
    if ( token_index == token_buf_size )
    {
      token_buf[ token_buf_size - 1 ] = 0;
      LOG_ERROR( main_logger, "KST token more than " << token_buf_size << " characters?  Partial token:\n'"
                 << token_buf << "'\n" );
      return make_pair("",false);
    }
    is.get( c );
    if ( ! is.good() ) break;

    // evaluate the chomping_whitespace status vs. current character
    if ( chomping_whitespace )
    {
      if (isspace( c ))
      {
        // we were chomping whitespace, and we see whitespace; do nothing
      }
      else
      {
        // we were chomping, but now we see non-whitespace; unset flag
        chomping_whitespace = false;
      }
    }

    // if we're still chomping whitespace, continue
    if ( chomping_whitespace ) continue;

    // no special characters if we're in the middle of a quoted string
    // (except the terminating quote)
    if (in_quotes)
    {
      token_buf[token_index++] = c;
      if ( c == '"' ) in_quotes = false;
      continue;
    }

    // if we're not in quotes, and we see a '"', enter in-quotes mode
    if (c == '"')
    {
      token_buf[token_index++] = c;
      in_quotes = true;
      continue;
    }

    // if it's a # (outside of quotes), read until either is.not-good or end-of-line
    if (c == '#')
    {
      char comment_c;
      bool in_comment = true;
      while (is.good() && in_comment)
      {
        is.get( comment_c );
        // end-of-line?
        if ((comment_c == '\n') || (comment_c == '\r'))
        {
          is.unget();
          in_comment = false;
        }
      }
      // set chomping-whitespace to true and loop back to top
      chomping_whitespace = true;
      continue;
    }

    // if it's a ';' or a ','; or whitespace, then we've completed the token
    if ((c == ',') || (c == ';') || isspace(c) ) break;

    // otherwise, add it to the token
    token_buf[token_index++] = c;

    // if it's a '[' or a ']', then we've completed the token
    if ((token_index == 1) &&
        ( (token_buf[0] == '[') || (token_buf[0] == ']')))
    {
      break;
    }

  } // ...while is.good()

  token_buf[token_index++] = 0;

  // if we broke out of the read loop chomping whitespace, the stream is done
  if (chomping_whitespace)
  {
    return make_pair( token_buf, false );
  }

  // otherwise, all is well
  return make_pair( token_buf, true );
}

void
parse_kst_node( istream& is,
                kst_node* this_node )
{
  static unsigned int depth = 0;
  string dbg_spacing;
  for (unsigned i=0; i<depth; ++i) dbg_spacing += "..";
  while (is.good())
  {
    kst_element* this_element = 0;
    pair< string, bool > pair = get_token( is );
    // all done if get_token says EOF
    if ( ! pair.second ) return;

    string s = pair.first;

    // do we descend a node?
    if (s == "[")
    {
      kst_node* sub_node = new kst_node();
      this_element = new kst_node_ptr( sub_node );
      ++depth;
      parse_kst_node( is, sub_node );
      --depth;
    }

    // are we finished with a node?
    else if ( s == "]" )
    {
      return;
    }

    // otherwise, it's a string; add it to the element list (only if it's nonempty)
    else if (s.size() > 0)
    {
      this_element = new kst_string( s );
    }

    else
    {

    }

    // add it to this node
    if ( this_element ) this_node->elist.push_back( this_element);

  } // while good
}

kst_node*
parse_kst_tree( istream& is )
{
  kst_node* this_node = new kst_node();
  parse_kst_node( is, this_node );
  return this_node;
}


} // anon namespace

namespace kwiver {
namespace track_oracle {

bool
file_format_kst
::inspect_file( const string& fn ) const
{
  ifstream is( fn.c_str() );
  if ( ! is )
  {
    LOG_ERROR( main_logger, "Couldn't open '" << fn << "'" );
    return false;
  }

  string line;
  if ( ! get_next_nonblank_line( is, line )) return false;
  istringstream iss( line );
  string tmp;
  if (!(iss >> tmp )) return false;
  return (tmp == "RESULTS,");
}

bool
file_format_kst
::read( const string& fn,
        track_handle_list_type& tracks ) const
{
  ifstream is( fn.c_str() );
  if ( ! is )
  {
    LOG_ERROR( main_logger, "Couldn't open '" << fn << "'" );
    return false;
  }

  LOG_INFO( main_logger, "KST: reading '" << fn << "'" );
  return this->read( is, tracks );
}


bool
file_format_kst
::read( istream& is,
        track_handle_list_type& tracks ) const
{
  vul_timer timer;
  kst_node* kst_tree = parse_kst_tree( is );
  unsigned n_toplevel_elements = kst_tree->elist.size();

  LOG_INFO( main_logger, "KST: Parsed " << n_toplevel_elements << " elements in "
            << timer.real() / 1000.0 << " seconds" );
  timer.mark();

  if ( n_toplevel_elements < 3 )
  {
    LOG_ERROR( main_logger, "KST tree only has " << n_toplevel_elements
               << " nodes; expecting at least 3" );
    return false;
  }

  unsigned index = 0;
  string p1, // file description
             p2; // file version number
  if ( ! ( kst_tree->e_is_str( index++, p1) && kst_tree->e_is_str( index++, p2 ) &&
           //(p1 == "RESULTS") && (p2 == "1" )))
           (p1 == "RESULTS") && (p2 == "2" )))
  {
    LOG_ERROR( main_logger, "KST version string is '" << p1 << "'; '" << p2 << "; expecting 'RESULTS 2'" );
    return false;
  }

  track_kst_type kst;

  // run down the top-level element list, extracting the track information
  //while (index + 11 < n_toplevel_elements )
  int node_element_count = 9;

#if 0
  for (size_t i=0; i<node_element_count; ++i)
  {
    string s("");
    bool rc = kst_tree->e_is_str( index+i, s);
    LOG_DEBUG( main_logger, "index " << index << " +i " << i << " : " << index+i << " : " << rc << " '" << s << "'" );
  }
#endif

  while (index + (node_element_count-1)  < n_toplevel_elements )
  {
    // parse the instance ID and scoring material

    int instance_id;
#if 0
    {
      string s("");
      bool rc = kst_tree->e_is_str( index+3, s);
      LOG_DEBUG( main_logger, "instance: index " << index << " +3 " << 3 << " : " << index+3 << " : " << rc << " '" << s << "'" );
    }
#endif
    if ( ! kst_tree->e_is_int( index+3, instance_id )) return false;

    vector<double> scores;
    double relevancy;
    unsigned rank;
    //if ( ! kst_tree->e_is_double( index+9, relevancy )) return false;
    if ( ! kst_tree->e_is_vector( index+7, scores )) return false;
    rank = static_cast<unsigned>(scores[0]);
    relevancy = scores[1];


    kst_node* n;
    //if ( ! kst_tree->e_is_node( index+11, n )) return false;
    if ( ! kst_tree->e_is_node( index+8, n )) return false;

    track_handle_type trk = kst.create();
    kst( trk ).instance_id() = instance_id;
    kst( trk ).relevancy() = relevancy;
    kst( trk ).rank() = rank;

    // move the index up
    //index += 12;
    //    index += 9;
    index += 10;

    // It turns out KSTs are not individual tracks, but *track groups.*
    // Hmm.  For scoring, we'd want the "merged result" sub-track, which
    // isn't there yet.  So in keeping with the sprit of the GUI :), for now,
    // take the first descriptor and score that.  For the moment, we read
    // in KSTs as tracks, not track groups.

    // Update: now search for classifiers down the elist.
    //
    // Further update: it seems PVMoving is reported as un-merged segments rather
    // than 'classifiers'.  Assume we'll see either a single classifier OR
    // multiple PVMoving segments, but not both.
    if ( ! n->elist.empty() )
    {
      kst_node* d;
      for (size_t node_index = 0; node_index<n->elist.size(); ++node_index)
      {
        if ( ! n->e_is_node( node_index, d )) return false;
        string s;
        if ( ! d->e_is_str( 0, s )) return false;
        bool is_classifier = ( (s == "\"Classifier\"") || (s == "\"classifier\"")); // that's great.
        bool is_pvmoving = ( s == "\"PersonOrVehicleMovement\"");
        if ( is_classifier || is_pvmoving )
        {
          kst_node* v_wrapper;
          if ( ! d->e_is_node( 4, v_wrapper )) return false;
          vector< double > classifier;
          if ( ! v_wrapper->e_is_vector( 0, classifier )) return false;

          // add classifier to schema
          kst( trk ).descriptor_classifier() = classifier;
        }
        else
        {
          continue;
        }

        kst_node* t_wrapper;
        if ( ! d->e_is_node( 5, t_wrapper )) return false;

        unsigned n_frames = t_wrapper->elist.size();
        for (unsigned f=0; f<n_frames; ++f)
        {
          kst_node* f_wrapper;
          if ( ! t_wrapper->e_is_node( f, f_wrapper )) return false;
          int frame_num;
          unsigned long long ts;
          vgl_box_2d<double> box;
          if ( ! f_wrapper->e_is_int( 0, frame_num )) return false;
          if ( ! f_wrapper->e_is_long_long( 1, ts )) return false;
          if ( ! f_wrapper->e_is_box( 2, box )) return false;

          // add frame_num, ts, box to schema
          frame_handle_type frame = kst.create_frame();
          kst[ frame ].frame_number() = frame_num;
          kst[ frame ].timestamp_usecs() = ts;
          kst[ frame ].bounding_box() = box;

        } // ...each frame

      } // ... classifier descriptor

    }
    tracks.push_back( trk );

  } // while more KST elements

  LOG_INFO( main_logger, "KST: Parsed " << tracks.size() << " tracks in "
            << timer.real()  / 1000.0 << " seconds" );
  timer.mark();
  // clean up
  delete kst_tree;
  LOG_INFO( main_logger, "KST: Cleaned up tree in " << timer.real() / 1000.0 << " seconds");

  // all done
  return true;

}

} // ...track_oracle
} // ...kwiver
