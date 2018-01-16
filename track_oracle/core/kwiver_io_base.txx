/*ckwg +5
 * Copyright 2014-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "kwiver_io_base.h"

#include <vector>
#include <tinyxml.h>

#include <track_oracle/core/kwiver_io_helpers.h>
#include <track_oracle/core/kwiver_io_base_data_io.h>

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t kib_logger( kwiver::vital::get_logger( __FILE__ ) );


using std::istream;
using std::vector;
using std::map;
using std::string;
using std::ostream;
using std::istringstream;

namespace // anon
{

typedef char (&no_io_type)[2];
typedef char yes_io_type;

struct proxy{ template <typename U> proxy(const U&); };

char tmp[2]; no_io_type not_used(tmp);
no_io_type operator>>( const proxy&, const proxy& ) { return not_used; }
no_io_type check( no_io_type ) { return not_used; }


template <typename U> yes_io_type check( const U& );


template< typename T >
class class_has_input_operator
{
  static istream& is;
  static T& val;

public:
  static const bool value = (sizeof(check(is >> val)) == sizeof( yes_io_type ));
};

template< bool T_has_input_operator, typename T >
struct input_handler
{
  static bool get( istream& is, T& val )
  {
    return static_cast<bool>( is >> val );
  }
};

template< typename T >
struct input_handler<false, T >
{
  static bool get( istream& is, T&  )
  {
    // If we get here, it's because from_str or vector operator>>
    // has been called on a type which (a) has no intrinsic op>>
    // and (b) does not have a overridden from_str.  So we
    // read a string to advance the stream and provide some sort
    // of output, and return false, and see what happens.
    //
    // An example of this is when track_vatic defined a set<string>
    // named "attributes" that we hadn't converted yet.

    string tmp( "failed-to-read" );
    is >> tmp;
    LOG_DEBUG( kib_logger, "kwiver_io_base input handler called for unimplemented type; string-read returned '" << tmp << "'" );
    return false;
  }
};

//
// If kwiver_io_helper implements a kwiver_write for T, use that,
// otherwise, use operator>>
//

template< typename T >
class class_has_kwiver_write
{
  static ostream& os;
public:
  static const bool value = (sizeof( ::kwiver::track_oracle::kwiver_write( os, T())) != sizeof(char));
};

template< bool T_has_kwiver_write, typename T>
struct output_handler
{
  static ostream& write( ostream& os, const T& val )
  {
    return ::kwiver::track_oracle::kwiver_write( os, val );
  }
};

template< typename T >
struct output_handler<false, T>
{
  static ostream& write( ostream& os, const T& val )
  {
    os << val;
    return os;
  }
};

template< typename T >
struct output_handler< false, vector<T> >
{
  static ostream& write( ostream& os, const vector<T>& vals )
  {
    for (size_t i=0; i<vals.size(); ++i)
    {
      output_handler< class_has_kwiver_write<T>::value, T>::write( os, vals[i] );
      os << " ";
    }
    return os;
  }
};

} // ...anon

namespace kwiver {
namespace track_oracle {

//
// generic vector read/write
//

template< typename T >
ostream& operator<<( ostream& os, const vector< T >& vals )
{
  for (size_t i=0; i<vals.size(); ++i)
  {
    output_handler< class_has_kwiver_write<T>::value, T>::write( os, vals[i] );
  }
  return os;
}

template< typename T >
istream& operator>>( istream& is, vector< T >& vals )
{
  T tmp;
  while ( input_handler< class_has_input_operator<T>::value, T >::get( is, tmp ))
  {
    vals.push_back( tmp );
  }
  return is;
}

template< typename DATA_TERM_T >
ostream&
kwiver_io_base<DATA_TERM_T>
::to_stream( ostream& os, const Type& val ) const
{
  //os << val;
  output_handler< class_has_kwiver_write<Type>::value, Type>::write( os, val );
  return os;
}

template< typename DATA_TERM_T >
bool
kwiver_io_base<DATA_TERM_T>
::from_str( const string& s, Type& val ) const
{
  istringstream iss( s );
  return input_handler< class_has_input_operator<Type>::value, Type>::get( iss, val );
}

template< typename DATA_TERM_T >
bool
kwiver_io_base<DATA_TERM_T>
::read_xml( const TiXmlElement* e, Type& val ) const
{
  if (! e->GetText()) return false;
  return this->from_str( e->GetText(), val );
}

template<typename DATA_TERM_T>
void
kwiver_io_base<DATA_TERM_T>
::write_xml( ostream& os, const string& indent, const Type& val ) const
{
  os << indent << "<" << this->name << "> ";
  this->to_stream( os, val );
  os << " </" << name << ">\n";
}

template<typename DATA_TERM_T>
vector< string>
kwiver_io_base<DATA_TERM_T>
::csv_headers() const
{
  vector< string > h;
  h.push_back( this->name );
  return h;
}

template<typename DATA_TERM_T>
bool
kwiver_io_base<DATA_TERM_T>
::from_csv( const map< string, string >& header_value_map, Type& val ) const
{
  const vector< string >& csv_h = this->csv_headers();
  string s("");
  for (size_t i=0; i<csv_h.size(); ++i)
  {
    map< string, string >::const_iterator p = header_value_map.find( csv_h[i] );
    if ( p == header_value_map.end() ) return false;
    s += p->second + " ";
  }
  return this->from_str( s, val );
}

template<typename DATA_TERM_T>
ostream&
kwiver_io_base<DATA_TERM_T>
::to_csv( ostream& os, const Type& val ) const
{
  return this->to_stream( os, val );
}

} // ...track_oracle
} // ...kwiver

