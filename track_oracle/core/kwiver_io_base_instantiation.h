// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_KWIVER_IO_BASE_INSTANTIATION_H
#define INCL_KWIVER_IO_BASE_INSTANTIATION_H

#include <track_oracle/core/kwiver_io_base.txx>

#define KWIVER_IO_BASE_INSTANCES(T) \
  template KWIVER_IO_EXPORT std::ostream& kwiver::track_oracle::kwiver_io_base<T>::to_stream( std::ostream& os, const Type& ) const; \
  template KWIVER_IO_EXPORT bool kwiver::track_oracle::kwiver_io_base<T>::from_str( const std::string& s, Type& val ) const; \
  template KWIVER_IO_EXPORT bool kwiver::track_oracle::kwiver_io_base<T>::read_xml( const TiXmlElement* e, Type& val ) const; \
  template KWIVER_IO_EXPORT void kwiver::track_oracle::kwiver_io_base<T>::write_xml( std::ostream& os, const std::string& indent, const Type& val ) const; \
  template KWIVER_IO_EXPORT std::vector<std::string> kwiver::track_oracle::kwiver_io_base<T>::csv_headers() const; \
  template KWIVER_IO_EXPORT bool kwiver::track_oracle::kwiver_io_base<T>::from_csv( const std::map< std::string, std::string>&, Type& ) const; \
  template KWIVER_IO_EXPORT std::ostream& kwiver::track_oracle::kwiver_io_base<T>::to_csv( std::ostream&, const Type& ) const;
#endif
