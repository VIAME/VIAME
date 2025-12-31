/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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

/**
 * \file
 * \brief Interface for read_object_track_set_db
 */

#ifndef VIAME_CPPDB_READ_OBJECT_TRACK_SET_DB_H
#define VIAME_CPPDB_READ_OBJECT_TRACK_SET_DB_H

#include <vital/vital_config.h>
#include "viame_cppdb_export.h"

#include <vital/algo/read_object_track_set.h>

#include <memory>

namespace viame {

class VIAME_CPPDB_EXPORT read_object_track_set_db
  : public kwiver::vital::algorithm_impl< read_object_track_set_db,
      kwiver::vital::algo::read_object_track_set >
{
public:
  read_object_track_set_db();
  virtual ~read_object_track_set_db();

  void set_configuration( kwiver::vital::config_block_sptr config ) override;
  bool check_configuration( kwiver::vital::config_block_sptr config ) const override;

  void open( std::string const& filename ) override;
  void close() override;
  bool read_set( kwiver::vital::object_track_set_sptr& set ) override;

private:
  class priv;
  std::unique_ptr< priv > d;
};

} // end namespace viame

#endif // VIAME_CPPDB_READ_OBJECT_TRACK_SET_DB_H
