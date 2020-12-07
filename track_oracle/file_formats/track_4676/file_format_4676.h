// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_FILE_FORMAT_4676_H
#define INCL_FILE_FORMAT_4676_H

#include "track_4676.h"

#include <track_oracle/file_format_base.h>

namespace vidtk
{

class file_format_4676: public file_format_base
{
public:
  file_format_4676(): file_format_base(TF_4676, "STANAG 4676 XML")
  {
    this->globs.push_back("*.xml");
    this->globs.push_back("*.4676");
  }
  virtual ~file_format_4676() {}

  virtual int supported_operations() const { return FF_READ_FILE; }

  // return a dynamically-allocated instance of the schema
  virtual track_base_impl* schema_instance() const { return new track_4676_type(); }

  // Inspect the file and return true if it is of this format
  virtual bool inspect_file(std::string const& fn) const;

  using file_format_base::read;
  // read tracks from the file
  virtual bool read(std::string const& fn,
                    track_handle_list_type& tracks) const;
};

} // vidtk

#endif
