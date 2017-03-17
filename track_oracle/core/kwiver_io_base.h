/*ckwg +5
 * Copyright 2014-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef INCL_KWIVER_IO_BASE_H
#define INCL_KWIVER_IO_BASE_H

///
/// Every track_oracle column delegates I/O to an instance of this class.
/// If the column comes from a data_term, that data_term supplies the
/// instance.  If it's an old-style ad hoc instance, for example:
///
///   track_field< double >( "latitude" );
///
/// ...then the element store creates one, using this base class.
///
/// The io_base is templated on either the data_term (if a data_term) or on
/// the ad hoc data type (if it's old-style.)  We rely on the data_term TMP
/// utils to correctly resolve DATA_TERM_T::Type.
///
/// CSV output happens in two phases: all the data terms are polled
/// for their headers, and then polled again on each row for their values.
///
/// CSV input reads the header line, then builds up a map of which data
/// terms claim each header.
///
/// A type may be represented by multiple headers, e.g. a box may have four
/// columns for its two corner points.  Thus the header output is a vector
/// of strings, and the input is a map of header-names to value strings.
/// By default, the map is re-written as a space-separated set of strings
/// and passed to the from_str() method.
///
///

#include <iostream>
#include <string>
#include <vector>
#include <map>

#include <track_oracle/data_terms/data_term_tmp_utils.h>

class TiXmlElement;

namespace kwiver {
namespace track_oracle {

template< typename T >
class kwiver_io_base
{
public:
  typedef typename data_term_traits< is_data_term< T >::value, T >::Type Type;
  explicit kwiver_io_base( const std::string& n ):
    name(n)
  {}
  virtual ~kwiver_io_base() {}
  virtual std::ostream& to_stream( std::ostream& os, const Type& val ) const;
  virtual bool from_str( const std::string& s,
                         Type& val ) const;
  virtual bool read_xml( const TiXmlElement* e,
                         Type& val ) const;
  virtual void write_xml( std::ostream& os,
                          const std::string& indent,
                          const Type& val ) const;
  virtual std::vector< std::string > csv_headers() const;
  virtual bool from_csv( const std::map< std::string, std::string >& header_value_map, Type& val ) const;
  virtual std::ostream& to_csv( std::ostream& os, const Type& val ) const;

protected:
  std::string name;

};

} // ...track_oracle
} // ...kwiver

#endif
