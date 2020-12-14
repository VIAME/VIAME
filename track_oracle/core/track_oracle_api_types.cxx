// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "track_oracle_api_types.h"

using std::istream;
using std::ostream;

namespace kwiver {
namespace track_oracle {

bool operator==( const frame_handle_type& lhs, const frame_handle_type& rhs ) { return lhs.row == rhs.row; }
bool operator==( const track_handle_type& lhs, const track_handle_type& rhs ) { return lhs.row == rhs.row; }
bool operator!=( const frame_handle_type& lhs, const frame_handle_type& rhs ) { return lhs.row != rhs.row; }
bool operator!=( const track_handle_type& lhs, const track_handle_type& rhs ) { return lhs.row != rhs.row; }
bool operator<( const track_handle_type& lhs, const track_handle_type& rhs ) { return lhs.row < rhs.row; }
bool operator<( const frame_handle_type& lhs, const frame_handle_type& rhs ) { return lhs.row < rhs.row; }
ostream& operator<<( ostream& os, const track_handle_type& t ) { os << "t:" << t.row;  return os; }
ostream& operator<<( ostream& os, const frame_handle_type& f ) { os << "f:" << f.row;  return os; }
istream& operator>>( istream& is, track_handle_type& t ) { char t1, t2; is >> t1 >> t2 >> t.row;  return is; }
istream& operator>>( istream& is, frame_handle_type& f ) { char t1, t2; is >> t1 >> t2 >> f.row;  return is; }

} // ...track_oracle
} // ...kwiver

