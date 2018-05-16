/*ckwg +29
 * Copyright 2013-2017 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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
 * \brief Header file for \link
 *        kwiver::arrows::database::query_track_descriptor_set_db
 *        query_track_descriptor_set_db \endlink
 */

#ifndef ARROWS_DATABASE_QUERY_TRACK_DESCRIPTOR_SET_DB_H_
#define ARROWS_DATABASE_QUERY_TRACK_DESCRIPTOR_SET_DB_H_

#include <vital/algo/query_track_descriptor_set.h>
#include <arrows/database/kwiver_algo_database_export.h>

#include <cppdb/frontend.h>

namespace kwiver {
namespace arrows {
namespace database {

class KWIVER_ALGO_DATABASE_EXPORT query_track_descriptor_set_db
  : public vital::algorithm_impl< query_track_descriptor_set_db,
      vital::algo::query_track_descriptor_set >
{
public:
  query_track_descriptor_set_db();
  virtual ~query_track_descriptor_set_db();

  virtual void set_configuration( vital::config_block_sptr config );
  virtual bool check_configuration( vital::config_block_sptr config ) const;
  virtual bool get_track_descriptor( std::string const& uid,
    desc_tuple_t& result );

  virtual void use_tracks_for_history( bool value );

protected:
  void connect_to_database_on_demand();

private:
  class priv;
  std::unique_ptr< priv > d;
};

} } }

#endif // ARROWS_DATABASE_QUERY_TRACK_DESCRIPTOR_SET_DB_H_
