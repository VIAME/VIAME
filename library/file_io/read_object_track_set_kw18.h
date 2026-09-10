// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Interface for read_object_track_set_kw18

#ifndef KWIVER_ARROWS_READ_OBJECT_TRACK_SET_KW18_H
#define KWIVER_ARROWS_READ_OBJECT_TRACK_SET_KW18_H

#include "viame_file_io_export.h"
#include <viame/algorithm_framework/vital_config.h>

#include <viame/algorithm_framework/algo/algorithm.txx>
#include <viame/algorithm_framework/algo/read_object_track_set.h>

#include <memory>

namespace kwiver {

namespace arrows {

namespace core {

class VIAME_FILE_IO_EXPORT read_object_track_set_kw18
  : public vital::algo::read_object_track_set
{
public:
  PLUGGABLE_IMPL(
    read_object_track_set_kw18,
    "Object track set kw18 reader.",
    PARAM_DEFAULT( delim, std::string, "delimeter", " " ),
    PARAM_DEFAULT( batch_load, bool, "batch_load", true ),
  )

  virtual ~read_object_track_set_kw18();

  virtual bool check_configuration( vital::config_block_sptr config ) const;

  virtual bool read_set( kwiver::vital::object_track_set_sptr& set );

private:
  void initialize() override;

  /// private implementation class
  class priv;
  KWIVER_UNIQUE_PTR( priv, d );
};

} // namespace core

} // namespace arrows

}     // end namespace

#endif // KWIVER_ARROWS_READ_OBJECT_TRACK_SET_KW18_H
