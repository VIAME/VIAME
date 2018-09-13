/*ckwg +5
 * Copyright 2012-2014 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "file_format_4676.h"

#include <track_oracle/xml_tokenizer.h>

#include <stanag_4676/stanag_4676.h>

#include <logger/logger.h>

#include <tinyxml.h>

#include <iostream>
#include <fstream>
#include <sstream>

using std::numeric_limits;
using std::vector;
using std::string;


VIDTK_LOGGER("file_format_4676");

namespace // anon
{

void
extract_frame(STANAG_4676::TrackPoint const* f,
              vidtk::track_4676_type& frame)
{
  // Read world location
  STANAG_4676::GeodeticPosition::sptr const pos = f->getPosition();
  frame.world_x() = pos->getLongitude();
  frame.world_y() = pos->getLatitude();
  double const z = pos->getElevation();
  if (z != numeric_limits<double>::min())
  {
    frame.world_z() = z;
  }

  // Set world location as vgl_point_3d, using '0' as the altitude
  // if we were unable to read it
  frame.world_location().set(
    frame.world_x(), frame.world_y(),
    frame.world_z.exists() ? frame.world_z() : 0.0);

  // Read image location, if specified
  STANAG_4676::TrackPointDetail::sptr detail = f->getDetail();
  if (detail)
  {
    STANAG_4676::Position::sptr const dp = detail->getPosition();
    if (dp && dp->toPixelPosition())
    {
      STANAG_4676::PixelPosition const* const pp = dp->toPixelPosition();
      frame.obj_x() = pp->getX();
      frame.obj_y() = pp->getY();
      frame.obj_location().set(frame.obj_x(), frame.obj_y());
    }
  }
}

bool
extract_track(STANAG_4676::Track::sptr t,
              vidtk::track_handle_list_type& tracks)
{
  vidtk::track_4676_type track;
  tracks.push_back(track.create());

  try
  {
    track.external_id() = t->getNumber().stoi();
  }
  catch (const std::invalid_argument&)
  {
    LOG_ERROR("Failed to convert track number '" << t->getNumber()
              << "'; ignoring track\n");
    return false;
  }

  track.unique_id() = t->getUUID();
  track.augmented_annotation() = t->getComment();

  boost::posix_time::ptime const& epoch = boost::posix_time::from_time_t(0);
  vector<STANAG_4676::TrackItem::sptr> const& items = t->getItems();
  size_t const k = items.size();
  for (size_t i = 0; i < k; ++i)
  {
    STANAG_4676::TrackPoint const* const f = items[i]->toTrackPoint();
    if (f)
    {
      vidtk::track_4676_type& frame = track[track.create_frame()];

      frame.timestamp_usecs() = (f->getTime() - epoch).total_microseconds();
      extract_frame(f, frame);
    }
  }

  return true;
}

bool
extract_tracks(STANAG_4676::TrackMessage::sptr message,
               vidtk::track_handle_list_type& tracks)
{
  vector<STANAG_4676::Track::sptr> const& trackPtrs =
    message->getTracks();
  size_t const k = trackPtrs.size();
  for (size_t i = 0; i < k; ++i)
  {
    if (!extract_track(trackPtrs[i], tracks))
      return false;
  }
  return true;
}

} // anon


namespace vidtk
{

bool
file_format_4676
::inspect_file(string const& fn) const
{
  vector< string > tokens = xml_tokenizer::first_n_tokens(fn, 10);
  bool has_product = false;
  bool has_stanag4676 = false;
  for (size_t i=0; i<tokens.size(); ++i)
  {
    if (tokens[i].find("<trackProduct") != string::npos) has_product = true;
    if (tokens[i].find("stanag4676") != string::npos) has_stanag4676 = true;
  }
  return has_product && has_stanag4676;
}

bool
file_format_4676
::read(string const& fn, track_handle_list_type& tracks) const
{
  // Load XML and parse into DOM
  LOG_INFO("TinyXML loading '" << fn << "': start");
  TiXmlDocument doc(fn.c_str());
  if (!doc.LoadFile())
  {
    LOG_ERROR("TinyXML (4676) couldn't load '" << fn << "'; skipping\n");
    return false;
  }
  LOG_INFO("TinyXML loading '" << fn << "': complete");

  TiXmlElement const* const xml_root = doc.RootElement();
  if (!xml_root)
  {
    LOG_ERROR("Couldn't load root element from '" << fn << "'; skipping\n");
    return false;
  }

  // Parse the DOM into a STANAG 4676 data tree
  STANAG_4676::TrackProductType::sptr const product =
    STANAG_4676::TrackProductType::fromXML(xml_root);
  if (!product)
  {
    LOG_ERROR("Failed to parse STANAG 4676 XML\n");
    return false;
  }

  return extract_tracks(product->getMessage(), tracks);
}

} // vidtk
