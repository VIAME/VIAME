/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Interface for read_detected_object_set_oceaneyes
 */

#ifndef VIAME_CORE_READ_DETECTED_OBJECT_SET_OCEANEYES_H
#define VIAME_CORE_READ_DETECTED_OBJECT_SET_OCEANEYES_H

#include "viame_core_export.h"

#include <vital/algo/detected_object_set_input.h>

#include <memory>

namespace viame {

class VIAME_CORE_EXPORT read_detected_object_set_oceaneyes
  : public kwiver::vital::algo::detected_object_set_input
{
public:
  static constexpr char const* name = "oceaneyes";

  // NOTE: Keep description in sync with write_detected_object_set_oceaneyes
  static constexpr char const* description =
    "Detected object set reader using oceaneyes csv format.\n\n"
    "  - filename, drop id, subject id, n, species identification,\n"
    "  - no fish confidence metric, yes fish confidence metric,\n"
    "  - species ID confidence metric, line confidence metric,\n"
    "  - is overmerged?, can see head/tail, head/tail coordinates";

  read_detected_object_set_oceaneyes();
  virtual ~read_detected_object_set_oceaneyes();

  virtual void set_configuration( kwiver::vital::config_block_sptr config );
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual bool read_set( kwiver::vital::detected_object_set_sptr& set,
                         std::string& image_name );

private:
  virtual void new_stream();

  class priv;
  std::unique_ptr< priv > d;
};

} // end namespace

#endif // VIAME_CORE_READ_DETECTED_OBJECT_SET_OCEANEYES_H
