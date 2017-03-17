/*ckwg +5
 * Copyright 2010-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef INCL_TRACK_FIELD_BASE_H
#define INCL_TRACK_FIELD_BASE_H

#include <vital/vital_config.h>
#include <track_oracle/core/track_oracle_export.h>

#include <track_oracle/core/track_oracle_core.h>
#include <track_oracle/core/track_field_host.h>
#include <iostream>

// This class contains the typeless methods related to column (field)
// access.  It maps the name of the data column to the column handle,
// but is stateless with respect to rows.
//
// Note that all actual data access is delegated to derived classes,
// which will have the type available so as to typecast the results.

namespace kwiver {
namespace track_oracle {

class track_field_host;
struct element_descriptor;

class TRACK_ORACLE_EXPORT track_field_base
{
  friend class track_oracle_row_view;
protected:
  explicit track_field_base( const std::string& n );
  explicit track_field_base( const std::string& n, track_field_host* h );
  std::string name;
  field_handle_type field_handle;
  track_field_host* host;

public:
  virtual ~track_field_base();
  std::string get_field_name() const;
  field_handle_type get_field_handle() const;
  virtual void remove_at_row( const oracle_entry_handle_type& row );
  virtual bool exists() const;
  virtual track_field_base* clone() const = 0;
  virtual void copy_value( const oracle_entry_handle_type& src,
                           const oracle_entry_handle_type& dst ) const = 0;
  void set_host( track_field_host* h );
};

} // ...track_oracle
} // ...kwiver

#endif
