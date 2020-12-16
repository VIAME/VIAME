// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <arrows/serialize/json/load_save.h>
#include <arrows/serialize/json/load_save_track_state.h>

#include <vital/internal/cereal/cereal.hpp>
#include <vital/internal/cereal/types/vector.hpp>
#include <vital/internal/cereal/archives/json.hpp>
#include <vital/internal/cereal/types/utility.hpp>

#include <sstream>
#include <iostream>

track_item::track_item()
{
  trk_sptr = kwiver::vital::track::create();
}

track_item::track_item(kwiver::vital::track_sptr& _trk_sptr )
{
  trk_sptr = _trk_sptr;
}

kwiver::vital::track_sptr& get_track()
{
  return trk_sptr;
}

