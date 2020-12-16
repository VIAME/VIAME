// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of kwiver::arrows::core::colorize functions to extract/compute colors
 */

#include "colorize.h"

namespace kwiver {
namespace arrows {
namespace core {

/// Extract feature colors from a frame image
vital::feature_set_sptr
extract_feature_colors(
  vital::feature_set const& features,
  vital::image_container const& image)
{
  const vital::image_of<uint8_t> image_data(image.get_image());
  std::vector<vital::feature_sptr> in_feat = features.features();
  std::vector<vital::feature_sptr> out_feat;
  out_feat.reserve(in_feat.size());

  for (auto const& f : in_feat)
  {
    auto const& loc = f->loc();
    auto const fd = std::make_shared<vital::feature_d>(*f);
    fd->set_color(image_data.at(static_cast<unsigned>(loc[0]),
                                static_cast<unsigned>(loc[1])));
    out_feat.push_back(fd);
  }

  return std::make_shared<vital::simple_feature_set>(out_feat);
}

/// Extract feature colors from a frame image
vital::feature_track_set_sptr
extract_feature_colors(
  vital::feature_track_set_sptr tracks,
  vital::image_container const& image,
  vital::frame_id_t frame_id)
{
  if (!tracks)
  {
    return nullptr;
  }
  const vital::image_of<uint8_t> image_data(image.get_image());

  for (auto& state : tracks->frame_states( frame_id ))
  {
    auto fts = std::dynamic_pointer_cast<vital::feature_track_state>(state);
    if ( !fts )
    {
      continue;
    }

    auto const feat = std::make_shared<vital::feature_d>(*fts->feature);
    auto const& loc = feat->get_loc();
    feat->set_color(image_data.at(static_cast<unsigned>(loc[0]),
                                  static_cast<unsigned>(loc[1])));

    fts->feature = feat;
  }

  return tracks;
}

/// Compute colors for landmarks
vital::landmark_map_sptr compute_landmark_colors(
  vital::landmark_map const& landmarks,
  vital::feature_track_set const& tracks)
{
  auto colored_landmarks = landmarks.landmarks();
  auto const no_such_landmark = colored_landmarks.end();

  for (auto const track : tracks.tracks())
  {
    auto const lmid = static_cast<vital::landmark_id_t>(track->id());
    auto lmi = colored_landmarks.find(lmid);
    if (lmi != no_such_landmark)
    {
      int ra = 0, ga = 0, ba = 0, k = 0; // accumulators
      for (auto const& ts : *track)
      {
        auto fts = std::dynamic_pointer_cast<vital::feature_track_state>(ts);
        if( !fts )
        {
          continue;
        }
        auto const& color = fts->feature->color();
        ra += color.r;
        ga += color.g;
        ba += color.b;
        ++k;
      }

      if (k)
      {
        auto const r = static_cast<unsigned char>(ra / k);
        auto const g = static_cast<unsigned char>(ga / k);
        auto const b = static_cast<unsigned char>(ba / k);

        auto lm = std::make_shared<kwiver::vital::landmark_d>(*(lmi->second));
        lm->set_color({r, g, b});
        lmi->second = lm;
      }
    }
  }

  return std::make_shared<kwiver::vital::simple_landmark_map>(colored_landmarks);
}

} // end namespace core
} // end namespace arrows
} // end namespace kwiver
