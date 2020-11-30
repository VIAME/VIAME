// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface for write_track_descriptor_set_csv
 */

#ifndef KWIVER_ARROWS_WRITE_TRACK_DESCRIPTOR_SET_CSV_H
#define KWIVER_ARROWS_WRITE_TRACK_DESCRIPTOR_SET_CSV_H

#include <vital/vital_config.h>
#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/algo/write_track_descriptor_set.h>

#include <memory>

namespace kwiver {
namespace arrows {
namespace core {

class KWIVER_ALGO_CORE_EXPORT write_track_descriptor_set_csv
  : public vital::algo::write_track_descriptor_set
{
public:
  PLUGIN_INFO( "csv",
               "Track descriptor set csv writer." )

  write_track_descriptor_set_csv();
  virtual ~write_track_descriptor_set_csv();

  virtual void set_configuration( vital::config_block_sptr config );
  virtual bool check_configuration( vital::config_block_sptr config ) const;

  virtual void write_set( const kwiver::vital::track_descriptor_set_sptr set );

private:
  class priv;
  std::unique_ptr< priv > d;
};

} } } // end namespace

#endif // KWIVER_ARROWS_TRACK_DESCRIPTOR_SET_OUTPUT_CSV_H
