/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Interface for read_object_track_set_viame_csv
 */

#ifndef VIAME_CORE_READ_OBJECT_TRACK_SET_VIAME_CSV_H
#define VIAME_CORE_READ_OBJECT_TRACK_SET_VIAME_CSV_H

#include "viame_core_export.h"

#include <vital/algo/read_object_track_set.h>

#include <memory>

namespace viame {

class VIAME_CORE_EXPORT read_object_track_set_viame_csv
  : public kwiver::vital::algo::read_object_track_set
{
public:

  static constexpr char const* name = "viame_csv";

  static constexpr char const* description =
    "Object track set viame_csv reader.\n\n"
    "  - Column(s) 1: Detection or Track ID\n"
    "  - Column(s) 2: Video or Image Identifier\n"
    "  - Column(s) 3: Unique Frame Identifier\n"
    "  - Column(s) 4-7: Img-bbox(TL_x,TL_y,BR_x,BR_y)"
    "  - Column(s) 8: Detection Confidence\n"
    "  - Column(s) 9: Target Length (0 or less if uncomputed)\n"
    "  - Column(s) 10-11+: Repeated Species, Confidence Pairs\n";

  read_object_track_set_viame_csv();
  virtual ~read_object_track_set_viame_csv();

  virtual void open( std::string const& filename );

  virtual void set_configuration( kwiver::vital::config_block_sptr config );
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual bool read_set( kwiver::vital::object_track_set_sptr& set );

private:
  class priv;
  std::unique_ptr< priv > d;
};

} // end namespace

#endif // VIAME_CORE_READ_OBJECT_TRACK_SET_VIAME_CSV_H
