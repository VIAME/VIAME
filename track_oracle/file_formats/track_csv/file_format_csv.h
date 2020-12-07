// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_FILE_FORMAT_CSV_H
#define INCL_FILE_FORMAT_CSV_H

#include <vital/vital_config.h>
#include <track_oracle/file_formats/track_csv/track_csv_export.h>

#include <track_oracle/file_formats/track_csv/track_csv.h>
#include <track_oracle/file_formats/file_format_base.h>

namespace kwiver {
namespace track_oracle {

class TRACK_CSV_EXPORT file_format_csv: public file_format_base
{
public:
  file_format_csv();
  virtual ~file_format_csv();

  virtual int supported_operations() const { return FF_READ | FF_WRITE; }

  // return a dynamically-allocated instance of the schema
  virtual track_base_impl* schema_instance() const { return new track_csv_type(); }

  // Inspect the file and return true if it is of this format
  virtual bool inspect_file(std::string const& fn) const;

  // read tracks from a file or stream
  virtual bool read(std::string const& fn,
                    track_handle_list_type& tracks) const;
  virtual bool read(std::istream& is,
                    track_handle_list_type& tracks) const;

  // write tracks to a file or stream
  virtual bool write(std::string const& fn,
                     track_handle_list_type const& tracks) const;
  virtual bool write(std::ostream& os,
                     track_handle_list_type const& tracks) const;

protected:
  bool internal_stream_read( std::istream& is, size_t file_size, track_handle_list_type& tracks ) const;
};

} // ...track_oracle
} // ...kwiver

#endif
