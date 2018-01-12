/*ckwg +5
 * Copyright 2012-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "track_field_output_specializations.h"

#include <stdexcept>
#include <utility>
#include <iostream>
#include <vector>
#include <sstream>

using std::ostream;
using std::pair;
using std::runtime_error;
using std::set;
using std::string;
using std::vector;
using std::map;

namespace kwiver {
namespace track_oracle {

// specialization for e.g. frame lists
template< >
ostream& operator<<( ostream& os,
                         const track_field< frame_handle_list_type >& f ) {
  os << " (" << f.field_handle << ") "
     << f.name;
  try
  {
    frame_handle_list_type frame_list = f();
    os << ": frames are: [ ";
    for (unsigned i=0; i<frame_list.size(); ++i) {
      os << frame_list[i].row << " ";
    }
    os << "]";
  }
  catch (runtime_error const& )
  {
    os << " (no row set)";
  }
  return os;
}

template< >
ostream& operator<<( ostream& os,
                         const track_field< track_handle_list_type >& f ) {
  os << " (" << f.field_handle << ") "
     << f.name;
  try
  {
    track_handle_list_type track_list = f();
    os << ": tracks are: [ ";
    for (unsigned i=0; i<track_list.size(); ++i) {
      os << track_list[i].row << " ";
    }
    os << "]";
  }
  catch (runtime_error const& )
  {
    os << " (no row set)";
  }
  return os;
}

template< >
ostream& operator<<( ostream& os,
                         const track_field< vector< unsigned int> >& f ) {
  os << " (" << f.field_handle << ") " << f.name;
  try
  {
    vector<unsigned> d = f();
    os << "[size=" << d.size() << "] ";
    for (unsigned i=0; i<d.size(); ++i)
    {
      os << d[i] << " ";
    }
  }
  catch (runtime_error const& )
  {
    os << " (no row set)";
  }
  return os;
}

template< >
ostream& operator<<( ostream& os,
                         const track_field< pair<unsigned int, unsigned int> >& f ) {
  os << " (" << f.field_handle << ") " << f.name;
  try
  {
    pair<unsigned, unsigned> d = f();
    os << "( " << d.first << ", " << d.second << ") ";
  }
  catch (runtime_error const& )
  {
    os << " (no row set)";
  }
  return os;
}


template< >
ostream& operator<<( ostream& os,
                         const track_field< vector< double> >& f ) {
  os << " (" << f.field_handle << ") " << f.name;
  try
  {
    vector<double> d = f();

    os << "[size=" << d.size() << "] ";
    for (unsigned i=0; i<d.size(); ++i)
    {
      os << d[i] << " ";
    }

  }
  catch (runtime_error const& )
  {
    os << " (no row set)";
  }
  return os;
}

template< >
ostream& operator<<( ostream& os,
                         const track_field< vector< vector<double> > >& f) {
  os << " (" << f.field_handle << ") " << f.name;
  try
  {
    vector< vector< double > > d = f();
    os << "[size = " << d.size() << "]\n";
    for (unsigned i=0; i<d.size(); ++i)
    {
      vector< double > row = d[i];
      os << "  " << i << " [ size = " << row.size() << "] \n    ";
      for (unsigned j=0; j<row.size(); ++j)
      {
        os << row[j] << " ";
      }
      os << "\n";
    }
  }
  catch (runtime_error const& )
  {
    os << " (no row set)";
  }
  return os;
}

template< >
ostream& operator<<( ostream& os,
                         const track_field< vector< string> >& f )
{
  os << " (" << f.field_handle << ") " << f.name;
  try
  {
    vector<string> d = f();
    os << "[size=" << d.size() << "] ";
    for (unsigned i=0; i<d.size(); ++i)
    {
      os << d[i] << " ";
    }
  }
  catch (runtime_error const& )
  {
    os << " (no row set)";
  }
  return os;
}

template< >
ostream& operator<<( ostream& os,
                         const track_field< set< string> >& f )
{
  os << " (" << f.field_handle << ") " << f.name;
  try
  {
    set<string> d = f();
    os << "[size=" << d.size() << "] ";
    set<string>::const_iterator iter, end = d.end();
    for (iter = d.begin(); iter != end; ++iter)
    {
      os << *iter << " ";
    }
  }
  catch (runtime_error const& )
  {
    os << " (no row set)";
  }
  return os;
}

template< >
ostream& operator<<( ostream& os,
                     const track_field< map< string, double > >& f )
{
  os << " (" << f.field_handle << ") " << f.name;
  return os;
}

} // ...track_oracle
} // ...kwiver

