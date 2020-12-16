// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef ARROWS_SERIALIZATION_JSON_TRACK_ITEM_H
#define ARROWS_SERIALIZATION_JSON_TRACK_ITEM_H

#include <arrows/serialize/json/load_save.h>
#include <arrows/serialize/json/load_save_track_state.h>

#include <vital/internal/cereal/cereal.hpp>
#include <vital/internal/cereal/types/vector.hpp>
#include <vital/internal/cereal/archives/json.hpp>
#include <vital/internal/cereal/types/utility.hpp>

#include <vital/logger/logger.h>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace json {

struct track_item
{
  kwiver::vital::track_sptr trk_sptr;
  
  track_item()
  {
    trk_sptr = kwiver::vital::track::create();
  }

  track_item(kwiver::vital::track_sptr& _trk_sptr )
  {
    trk_sptr = _trk_sptr;
  }

  kwiver::vital::track_sptr& get_track()
  {
    return trk_sptr;
  }

  template<class Archive>
  void save ( Archive& archive ) const
  {
    archive( cereal::make_nvp( "track_id", trk_sptr->id() ) );
    archive( cereal::make_nvp( "track_size", trk_sptr->size() ) );
    std::vector<kwiver::vital::track_state_sptr> trk;
    for ( auto trk_state_itr=trk_sptr->begin(); trk_state_itr!=trk_sptr->end();
              ++trk_state_itr)
    {
      auto trk_state = *trk_state_itr;
      trk.push_back(trk_state);
    }
    archive(cereal::make_nvp( "trk", trk) );
  }

  template<class Archive>
  void load ( Archive& archive )
  {
    size_t track_size;
    kwiver::vital::track_id_t track_id;
    archive( CEREAL_NVP( track_size ) );
    archive( CEREAL_NVP( track_id) );
    std::vector<kwiver::vital::track_state_sptr> trk;
    archive( CEREAL_NVP( trk ));
    trk_sptr->set_id(track_id);
    for (auto trk_state : trk)
    {
      bool trk_inserted = trk_sptr->insert(trk_state);
      if ( !trk_inserted )
      {
        LOG_ERROR( kwiver::vital::get_logger( "data_serializer" ),
                 "Failed to insert track state in track" );
      }
    }
  }
};
  
} } } }       // end namespace kwiver

#endif // ARROWS_SERIALIZATION_JSON_TRACK_ITEM_H
