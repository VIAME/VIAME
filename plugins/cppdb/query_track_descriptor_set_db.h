/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Header file for \link
 *        kwiver::arrows::database::query_track_descriptor_set_db
 *        query_track_descriptor_set_db \endlink
 */

#ifndef VIAME_CPPDB_QUERY_TRACK_DESCRIPTOR_SET_DB_H
#define VIAME_CPPDB_QUERY_TRACK_DESCRIPTOR_SET_DB_H

#include <vital/algo/query_track_descriptor_set.h>
#include "viame_cppdb_export.h"

#include <cppdb/frontend.h>

namespace viame {

class VIAME_CPPDB_EXPORT query_track_descriptor_set_db
  : public kwiver::vital::algorithm_impl< query_track_descriptor_set_db,
      kwiver::vital::algo::query_track_descriptor_set >
{
public:
  query_track_descriptor_set_db();
  virtual ~query_track_descriptor_set_db();

  virtual void set_configuration( kwiver::vital::config_block_sptr config );
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;
  virtual bool get_track_descriptor( std::string const& uid,
    desc_tuple_t& result );

  virtual void use_tracks_for_history( bool value );

protected:
  void connect_to_database_on_demand();

private:
  class priv;
  std::unique_ptr< priv > d;
};

} // end namespace viame

#endif // VIAME_CPPDB_QUERY_TRACK_DESCRIPTOR_SET_DB_H
