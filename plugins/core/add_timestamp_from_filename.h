/*ckwg +29
 * Copyright 2019 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef VIAME_CORE_ADD_TIMESTAMP_FROM_FILENAME_H
#define VIAME_CORE_ADD_TIMESTAMP_FROM_FILENAME_H

#include <plugins/core/viame_core_export.h>

#include <vital/algo/image_io.h>

namespace viame
{

class VIAME_CORE_EXPORT add_timestamp_from_filename
  : public kwiver::vital::algo::image_io
{
public:
  static constexpr char const* name = "add_timestamp_from_filename";
  static constexpr char const* description =
    "Parse timestamps from an image filename when reading an image";

  add_timestamp_from_filename();
  ~add_timestamp_from_filename() override = default;

  kwiver::vital::config_block_sptr get_configuration() const override;

  void set_configuration( kwiver::vital::config_block_sptr config ) override;

  bool check_configuration( kwiver::vital::config_block_sptr config ) const override;

private:
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
