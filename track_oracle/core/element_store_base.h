// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_ELEMENT_STORE_BASE_H
#define INCL_ELEMENT_STORE_BASE_H

#include <vital/vital_config.h>
#include <track_oracle/core/track_oracle_export.h>

///
/// The base class for data columns in track_oracle.
/// Serves several purposes:
/// (1) holds the element_descriptor
/// (2) serves as the base class for actual storage so the exists() API
/// can be available without needing the type
/// (3) Provides the XML output prototype
/// (4) Provides the kwiver i/o interface

#include <track_oracle/core/track_oracle_api_types.h>
#include <track_oracle/core/element_descriptor.h>

class  TiXmlElement;

namespace kwiver {
namespace track_oracle {

class TRACK_ORACLE_EXPORT element_store_base
{
public:

  explicit element_store_base( const element_descriptor& ed ): d(ed) {}
  virtual ~element_store_base();

  const element_descriptor& get_descriptor() const;
  virtual bool exists( const track_handle_type& h ) const = 0;
  virtual bool exists( const frame_handle_type& h ) const = 0;
  virtual bool exists( const oracle_entry_handle_type& h ) const = 0;
  virtual std::vector< bool > exists( const std::vector< oracle_entry_handle_type >& sorted_hlist ) const = 0;

  virtual bool copy_value( const oracle_entry_handle_type& src, const oracle_entry_handle_type& dst ) = 0;
  virtual std::ostream& emit_as_kwiver( std::ostream& os, const oracle_entry_handle_type& h, const std::string& indent ) const = 0;
  virtual std::ostream& emit_as_csv( std::ostream& os, const oracle_entry_handle_type& h, bool emit_default_if_missing = false) const = 0;
  virtual bool read_kwiver_xml_to_row( const oracle_entry_handle_type& h, const TiXmlElement* e ) = 0;
  virtual bool read_csv_to_row( const oracle_entry_handle_type& h, const std::map< std::string, std::string>& header_value_map ) = 0;
  virtual std::vector<std::string> csv_headers() const = 0;
  virtual void set_to_default_value( const oracle_entry_handle_type& h ) = 0;

  virtual bool remove( const track_handle_type& h ) = 0;
  virtual bool remove( const frame_handle_type& h ) = 0;
  virtual bool remove( const oracle_entry_handle_type& h ) = 0;

private:
  element_descriptor d;
};

} // ...track_oracle
} // ...kwiver

#endif
