// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface for read_object_track_set_kw18
 */

#ifndef KWIVER_ARROWS_READ_OBJECT_TRACK_SET_KW18_H
#define KWIVER_ARROWS_READ_OBJECT_TRACK_SET_KW18_H

#include <vital/vital_config.h>
#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/algo/read_object_track_set.h>

#include <memory>

namespace kwiver {
namespace arrows {
namespace core {

class KWIVER_ALGO_CORE_EXPORT read_object_track_set_kw18
  : public vital::algo::read_object_track_set
{
public:
  PLUGIN_INFO( "kw18",
               "Object track set kw18 reader." )

  read_object_track_set_kw18();
  virtual ~read_object_track_set_kw18();

  virtual void set_configuration( vital::config_block_sptr config );
  virtual bool check_configuration( vital::config_block_sptr config ) const;

  virtual bool read_set( kwiver::vital::object_track_set_sptr& set );

private:
  class priv;
  std::unique_ptr< priv > d;
};

} } } // end namespace

#endif // KWIVER_ARROWS_READ_OBJECT_TRACK_SET_KW18_H
