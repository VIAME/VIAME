/*ckwg +5
 * Copyright 2014 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef INCL_KWIVER_IO_BASE_INSTANTIATION_H
#define INCL_KWIVER_IO_BASE_INSTANTIATION_H

#include <track_oracle/kwiver_io_base.txx>

#define KWIVER_IO_BASE_INSTANCES(T) \
  template std::ostream& kwiver::track_oracle::kwiver_io_base<T>::to_stream( std::ostream& os, const Type& ) const; \
  template bool kwiver::track_oracle::kwiver_io_base<T>::from_str( const std::string& s, Type& val ) const; \
  template bool kwiver::track_oracle::kwiver_io_base<T>::read_xml( const TiXmlElement* e, Type& val ) const; \
  template void kwiver::track_oracle::kwiver_io_base<T>::write_xml( std::ostream& os, const std::string& indent, const Type& val ) const; \
  template std::vector<std::string> kwiver::track_oracle::kwiver_io_base<T>::csv_headers() const; \
  template bool kwiver::track_oracle::kwiver_io_base<T>::from_csv( const std::map< std::string, std::string>&, Type& ) const; \
  template std::ostream& kwiver::track_oracle::kwiver_io_base<T>::to_csv( std::ostream&, const Type& ) const;
#endif
