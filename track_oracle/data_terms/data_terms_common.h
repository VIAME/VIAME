/*ckwg +5
 * Copyright 2014-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef INCL_DATA_TERMS_COMMON_H
#define INCL_DATA_TERMS_COMMON_H

///
/// A data term represents the unique semantic concept contained in a
/// track_oracle column.  If multiple file formats use the same data
/// term, say an image bounding box, they thereby assert that the box
/// means the same thing in both formats.
///
/// Since the goal of track_oracle is to enable dynamic composition of
/// data structures using data terms, it's important that the data
/// term be semantically self-contained.  If there are two ways to
/// interpret the image box in a file format, i.e. coordinates relative
/// to the image vs. relative to the AOI, these need to be expressed
/// via two separate but complete data terms ("image absolute box" and
/// "aoi relative box"), rather than two dependent terms ("box" and
/// "coordinate system"), because a downstream user might use "box" but
/// know nothing about "coordinate system".
///
/// A data term has two attributes:
///
/// - the name (must be unique)
///
/// - the C++ data type
///
/// Although data terms are declared in C++ namespaces, those namespaces
/// are not necessarily reflected in the data term's name member.
///
/// The intent is that a data term type may be used to create a data_field.
/// Thus no instance of the data term is required, thus the name must be
/// statically defined.  (Which is tedious.)
///
///

#include <string>
#include <track_oracle/data_terms/data_term_tmp_utils.h>

#include <vital/vital_config.h>
#include <track_oracle/data_terms/data_terms_export.h>

#include <track_oracle/core/track_oracle_api_types.h>
#include <track_oracle/core/kwiver_io_base.h>
#include <track_oracle/core/track_oracle_api_types.h>

#define DECL_DT(NAME, TYPE, DESC )         \
  struct DATA_TERMS_EXPORT NAME : public data_term_base, kwiver_io_base<TYPE>    \
{ \
  NAME(): kwiver::track_oracle::kwiver_io_base<TYPE>( #NAME ) {}                        \
  ~NAME() {} \
  typedef TYPE Type; \
  static context c; \
  static std::string get_context_name() { return #NAME; }   \
  static std::string get_context_description() { return #DESC; }         \
}

#define DECL_DT_DEFAULT(NAME, TYPE, DEFAULT, DESC )                      \
  struct DATA_TERMS_EXPORT NAME : public data_term_base, kwiver_io_base<TYPE>    \
{ \
  NAME(): kwiver::track_oracle::kwiver_io_base<TYPE>( #NAME ) {}                        \
  ~NAME() {} \
  typedef TYPE Type; \
  static TYPE get_default_value() { return DEFAULT; } \
  static context c; \
  static std::string get_context_name() { return #NAME; }   \
  static std::string get_context_description() { return #DESC; }         \
}

#define DECL_DT_RW_STRCSV(NAME, TYPE, DESC )         \
  struct DATA_TERMS_EXPORT NAME : public data_term_base, kwiver_io_base<TYPE> \
{ \
  NAME(): kwiver::track_oracle::kwiver_io_base<TYPE>( #NAME ) {}                        \
  ~NAME() {} \
  typedef TYPE Type; \
  static context c; \
  static std::string get_context_name() { return #NAME; }   \
  static std::string get_context_description() { return #DESC; }        \
  virtual std::ostream& to_stream( std::ostream& os, const Type& val ) const; \
  virtual bool from_str( const std::string& s, Type& val ) const;             \
  virtual std::vector< std::string > csv_headers() const; \
  virtual bool from_csv( const std::map< std::string, std::string >& header_value_map, Type& val ) const; \
  virtual std::ostream& to_csv( std::ostream& os, const Type& val ) const; \
}

#define DECL_DT_RW_STRCSV_DEFAULT(NAME, TYPE, DEFAULT, DESC )         \
  struct DATA_TERMS_EXPORT NAME : public data_term_base, kwiver_io_base<TYPE> \
{ \
  NAME(): kwiver::track_oracle::kwiver_io_base<TYPE>( #NAME ) {}                        \
  ~NAME() {} \
  typedef TYPE Type; \
  static TYPE get_default_value() { return DEFAULT; } \
  static context c; \
  static std::string get_context_name() { return #NAME; }   \
  static std::string get_context_description() { return #DESC; }        \
  virtual std::ostream& to_stream( std::ostream& os, const Type& val ) const; \
  virtual bool from_str( const std::string& s, Type& val ) const;             \
  virtual std::vector< std::string > csv_headers() const; \
  virtual bool from_csv( const std::map< std::string, std::string >& header_value_map, Type& val ) const; \
  virtual std::ostream& to_csv( std::ostream& os, const Type& val ) const; \
}

#define DECL_DT_W_STR(NAME, TYPE, DESC )         \
  struct DATA_TERMS_EXPORT NAME : public data_term_base, kwiver_io_base<TYPE> \
{ \
  NAME(): kwiver::track_oracle::kwiver_io_base<TYPE>( #NAME ) {}                        \
  ~NAME() {} \
  typedef TYPE Type; \
  static context c; \
  static std::string get_context_name() { return #NAME; }   \
  static std::string get_context_description() { return #DESC; }        \
  virtual std::ostream& to_stream( std::ostream& os, const Type& val ) const; \
}

#define DECL_DT_RW_STR(NAME, TYPE, DESC )         \
  struct DATA_TERMS_EXPORT NAME : public data_term_base, kwiver_io_base<TYPE> \
{ \
  NAME(): kwiver::track_oracle::kwiver_io_base<TYPE>( #NAME ) {}                        \
  ~NAME() {} \
  typedef TYPE Type; \
  static context c; \
  static std::string get_context_name() { return #NAME; }   \
  static std::string get_context_description() { return #DESC; }        \
  virtual std::ostream& to_stream( std::ostream& os, const Type& val ) const; \
  virtual bool from_str( const std::string& s, Type& val ) const;             \
}

#define DECL_DT_W_STRCSV(NAME, TYPE, DESC )         \
  struct DATA_TERMS_EXPORT NAME : public data_term_base, kwiver_io_base<TYPE> \
{ \
  NAME(): kwiver::track_oracle::kwiver_io_base<TYPE>( #NAME ) {}                        \
  ~NAME() {} \
  typedef TYPE Type; \
  static context c; \
  static std::string get_context_name() { return #NAME; }   \
  static std::string get_context_description() { return #DESC; }        \
  virtual std::ostream& to_stream( std::ostream& os, const Type& val ) const; \
  virtual std::vector< std::string > csv_headers() const; \
  virtual bool from_csv( const std::map< std::string, std::string >& header_value_map, Type& val ) const; \
  virtual std::ostream& to_csv( std::ostream& os, const Type& val ) const; \
}

#define DECL_DT_RW_STRXMLCSV(NAME, TYPE, DESC )         \
  struct DATA_TERMS_EXPORT NAME : public data_term_base, kwiver_io_base<TYPE> \
{ \
  NAME(): kwiver::track_oracle::kwiver_io_base<TYPE>( #NAME ) {}                        \
  ~NAME() {} \
  typedef TYPE Type; \
  static context c; \
  static std::string get_context_name() { return #NAME; }   \
  static std::string get_context_description() { return #DESC; }        \
  virtual std::ostream& to_stream( std::ostream& os, const Type& val ) const;  \
  virtual bool from_str( const std::string& s, Type& val ) const;             \
  virtual bool read_xml( const TiXmlElement* e, Type& val ) const;            \
  virtual void write_xml( std::ostream& os, \
                          const std::string& indent, \
                          const Type& val ) const; \
  virtual std::vector< std::string > csv_headers() const; \
  virtual bool from_csv( const std::map< std::string, std::string >& header_value_map, Type& val ) const; \
  virtual std::ostream& to_csv( std::ostream& os, const Type& val ) const; \
}

#define DECL_DT_RW_STRXML(NAME, TYPE, DESC )         \
  struct DATA_TERMS_EXPORT NAME : public data_term_base, kwiver_io_base<TYPE> \
{ \
  NAME(): kwiver::track_oracle::kwiver_io_base<TYPE>( #NAME ) {}                        \
  ~NAME() {} \
  typedef TYPE Type; \
  static context c; \
  static std::string get_context_name() { return #NAME; }   \
  static std::string get_context_description() { return #DESC; }        \
  virtual std::ostream& to_stream( std::ostream& os, const Type& val ) const;  \
  virtual bool from_str( const std::string& s, Type& val ) const;             \
  virtual bool read_xml( const TiXmlElement* e, Type& val ) const;            \
  virtual void write_xml( std::ostream& os, \
                          const std::string& indent, \
                          const Type& val ) const; \
}

#define DECL_DT_XMLCSV(NAME, TYPE, DESC )         \
  struct DATA_TERMS_EXPORT NAME : public data_term_base, kwiver_io_base<TYPE> \
{ \
  NAME(): kwiver::track_oracle::kwiver_io_base<TYPE>( #NAME ) {}                        \
  ~NAME() {} \
  typedef TYPE Type; \
  static context c; \
  static std::string get_context_name() { return #NAME; }   \
  static std::string get_context_description() { return #DESC; }        \
  virtual bool read_xml( const TiXmlElement* e, Type& val ) const;            \
  virtual void write_xml( std::ostream& os, \
                          const std::string& indent, \
                          const Type& val ) const; \
  virtual std::vector< std::string > csv_headers() const; \
  virtual bool from_csv( const std::map< std::string, std::string >& header_value_map, Type& val ) const; \
  virtual std::ostream& to_csv( std::ostream& os, const Type& val ) const; \
}

#endif
