// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * @file
 * @brief The track_oracle file format interface.
 *
 *
 */

#ifndef KWIVER_TRACK_ORACLE_FILE_FORMAT_KPF_H_
#define KWIVER_TRACK_ORACLE_FILE_FORMAT_KPF_H_

#include <vital/vital_config.h>
#include <track_oracle/file_formats/track_kpf_geom/track_kpf_geom_export.h>

#include <track_oracle/file_formats/track_kpf_geom/track_kpf_geom.h>
#include <track_oracle/file_formats/file_format_base.h>

namespace kwiver {
namespace track_oracle {

class TRACK_KPF_GEOM_EXPORT file_format_kpf_geom: public file_format_base
{
public:
  file_format_kpf_geom();
  virtual ~file_format_kpf_geom();

  virtual int supported_operations() const { return FF_READ | FF_WRITE; }

  // return a dynamically-allocated instance of the schema
  virtual track_base_impl* schema_instance() const { return new track_kpf_geom_type(); }

  // Inspect the file and return true if it is of this format
  virtual bool inspect_file(const std::string& fn) const;

  // read tracks from a file or stream
  virtual bool read( const std::string& fn,
                    track_handle_list_type& tracks) const;
  virtual bool read( std::istream& is,
                    track_handle_list_type& tracks) const;

  // write tracks to a file or stream
  virtual bool write( const std::string& fn,
                      const track_handle_list_type& tracks) const;
  virtual bool write( std::ostream& os,
                      const track_handle_list_type& tracks) const;

};

} // ...track_oracle
} // ...kwiver

#endif
