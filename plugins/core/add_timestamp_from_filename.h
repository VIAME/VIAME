/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_CORE_ADD_TIMESTAMP_FROM_FILENAME_H
#define VIAME_CORE_ADD_TIMESTAMP_FROM_FILENAME_H

#include "viame_core_export.h"

#include <vital/algo/image_io.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

namespace viame
{

class VIAME_CORE_EXPORT add_timestamp_from_filename
  : public kwiver::vital::algo::image_io
{
public:
  PLUGGABLE_IMPL(
    add_timestamp_from_filename,
    "Parse timestamps from an image filename when reading an image" )

  ~add_timestamp_from_filename() override = default;

  bool check_configuration( kwiver::vital::config_block_sptr config ) const override;

private:
  void initialize() override;

  void set_configuration_internal(
    kwiver::vital::config_block_sptr config ) override;

  kwiver::vital::config_block_sptr get_configuration() const override;

  kwiver::vital::algo::image_io_sptr image_reader;

  kwiver::vital::image_container_sptr load_(
    std::string const& filename ) const override;

  void save_( std::string const& filename,
    kwiver::vital::image_container_sptr data ) const override;

  kwiver::vital::metadata_sptr load_metadata_(
    std::string const& filename ) const override;

  kwiver::vital::metadata_sptr fixup_metadata(
    std::string const& filename, kwiver::vital::metadata_sptr md ) const;
};

} // end namespace viame

#endif // VIAME_CORE_ADD_TIMESTAMP_FROM_FILENAME_H
