#include <vital/types/timestamp.h>
#include "ocv_pair_stereo_tracks.h"
#include "ocv_pair_stereo_detections.h"

viame::ocv_pair_stereo_tracks::ocv_pair_stereo_tracks()
    : m_detection_pairing(new ocv_pair_stereo_detections()) {}

void viame::ocv_pair_stereo_tracks::load_camera_calibration() {
  m_detection_pairing->m_cameras_directory = m_cameras_directory;
  m_detection_pairing->load_camera_calibration();
}

std::tuple<std::vector<kwiver::vital::track_sptr>, std::vector<viame::Detections3DPositions>>
viame::ocv_pair_stereo_tracks::update_left_tracks_3d_position(
    const std::vector<kwiver::vital::track_sptr> &tracks, const cv::Mat &cv_disparity_map,
    const kwiver::vital::timestamp &timestamp) {
  m_detection_pairing->m_verbose = m_verbose;
  const auto cv_pos_3d_map = m_detection_pairing->reproject_3d_depth_map(cv_disparity_map);

  std::vector<kwiver::vital::track_sptr> filtered_tracks;
  std::vector<Detections3DPositions> tracks_positions;

  for (const auto &track: tracks) {
    // Check if a 3d track already exists with this id in order to update it
    // instead of generating a new track
    kwiver::vital::track_sptr tracks_3d;
    auto t_ptr = m_tracks_with_3d_left.find(track->id());
    if (t_ptr != m_tracks_with_3d_left.end()) {
      tracks_3d = t_ptr->second;
    } else {
      tracks_3d = track;
      m_tracks_with_3d_left[track->id()] = tracks_3d;
    }

    // Add 3D track to output
    filtered_tracks.push_back(tracks_3d);

    // Skip 3d processing for tracks without current frame and associate empty position to avoid further pairing
    auto state = std::dynamic_pointer_cast<kwiver::vital::object_track_state>(track->back());
    if ((track->last_frame() < timestamp.get_frame()) || !state) {
      tracks_positions.emplace_back(Detections3DPositions{});
      continue;
    }

    // Process 3D coordinates for frame matching the current depth image
    auto position = m_detection_pairing->update_left_detection_3d_position(state->detection(), cv_pos_3d_map);

    // Update state information to tracks 3D and push tracks to output
    tracks_3d->append(state);
    tracks_positions.emplace_back(position);
  }

  // Sanity check on the filtered_tracks and track_position size
  if (filtered_tracks.size() != tracks_positions.size())
    VITAL_THROW(kwiver::vital::invalid_data,
                "Expected same tracks and position size. Got : " + std::to_string(filtered_tracks.size()) + " VS " +
                std::to_string(tracks_positions.size()));

  return {filtered_tracks, tracks_positions};
}

std::vector<kwiver::vital::track_sptr> viame::ocv_pair_stereo_tracks::keep_right_tracks_in_current_frame(
    const std::vector<kwiver::vital::track_sptr> &tracks, const kwiver::vital::timestamp &timestamp) {
  std::vector<kwiver::vital::track_sptr> filtered_tracks;

  for (const auto &track: tracks) {
    // Get corresponding track in right track map or add track to map if not present yet
    if (m_right_tracks_memo.find(track->id()) == std::end(m_right_tracks_memo))
      m_right_tracks_memo[track->id()] = track;

    // Get track from dict (track present in dict contains updated ID
    auto paired_track = m_right_tracks_memo[track->id()];
    filtered_tracks.push_back(paired_track);

    // Skip tracks without state or which frames don't match current frame
    auto state = std::dynamic_pointer_cast<kwiver::vital::object_track_state>(track->back());
    if (((track->last_frame() < timestamp.get_frame()) || !state)) {
      continue;
    }

    // Update state in paired_track
    paired_track->append(state);
  }

  return filtered_tracks;
}


kwiver::vital::track_id_t viame::ocv_pair_stereo_tracks::last_left_right_track_id() const {
  auto get_map_ids = [](const std::map<kwiver::vital::track_id_t, kwiver::vital::track_sptr> &map) {
    std::set<kwiver::vital::track_id_t> track_ids;
    for (const auto &track: map) {
      track_ids.emplace(track.second->id());
    }
    return track_ids;
  };

  auto ids_left = get_map_ids(m_tracks_with_3d_left);
  auto ids_right = get_map_ids(m_right_tracks_memo);
  if (ids_left.empty() || ids_right.empty())
    return ids_left.empty() ? ids_right.empty() ? 1 : *ids_right.rbegin() + 1 : *ids_left.rbegin() + 1;

  return std::max(*ids_left.rbegin(), *ids_right.rbegin());
}

/// @brief Helper structure to store the most likely pair to a left track
struct MostLikelyPair {
  int frame_count{-1};
  kwiver::vital::track_id_t right_id{-1};
};


std::tuple<std::vector<kwiver::vital::track_sptr>, std::vector<kwiver::vital::track_sptr>>
viame::ocv_pair_stereo_tracks::get_left_right_tracks_with_pairing() {
  std::vector<kwiver::vital::track_sptr> left_tracks, right_tracks;

  auto proc_tracks = m_do_split_detections ? split_paired_tracks_to_new_tracks(left_tracks, right_tracks)
                                           : select_most_likely_pairing(left_tracks, right_tracks);

  // Append other tracks
  const auto append_unprocessed_tracks = [](
      const std::map<kwiver::vital::track_id_t, kwiver::vital::track_sptr> &tracks_map,
      std::set<kwiver::vital::track_id_t> &processed, std::vector<kwiver::vital::track_sptr> &vect) {
    for (const auto &pair: tracks_map) {
      if (processed.find(pair.first) != std::end(processed))
        continue;

      processed.emplace(pair.first);
      vect.emplace_back(pair.second->clone());
    }
  };

  append_unprocessed_tracks(m_tracks_with_3d_left, std::get<0>(proc_tracks), left_tracks);
  append_unprocessed_tracks(m_right_tracks_memo, std::get<1>(proc_tracks), right_tracks);

  return {filter_tracks_with_threshold(left_tracks), filter_tracks_with_threshold(right_tracks)};
}

std::tuple<std::set<kwiver::vital::track_id_t>, std::set<kwiver::vital::track_id_t>>
viame::ocv_pair_stereo_tracks::select_most_likely_pairing(std::vector<kwiver::vital::track_sptr> &left_tracks,
                                                                    std::vector<kwiver::vital::track_sptr> &right_tracks) {
  std::map<kwiver::vital::track_id_t, MostLikelyPair> most_likely_left_to_right_pair;
  // Find the most likely pair from all pairs across all frames
  for (const auto &pair: m_left_to_right_pairing) {
    const auto id_pair = pair.second.left_right_id_pair;
    if (most_likely_left_to_right_pair.find(id_pair.left_id) == std::end(most_likely_left_to_right_pair))
      most_likely_left_to_right_pair[id_pair.left_id] = MostLikelyPair{};

    const auto pair_frame_count = (int) pair.second.frame_set.size();
    if (pair_frame_count > most_likely_left_to_right_pair[id_pair.left_id].frame_count) {
      most_likely_left_to_right_pair[id_pair.left_id].frame_count = (int) pair_frame_count;
      most_likely_left_to_right_pair[id_pair.left_id].right_id = id_pair.right_id;
    }
  }

  // Iterate over all paired tracks first
  std::set<kwiver::vital::track_id_t> proc_left, proc_right;
  auto last_track_id = last_left_right_track_id() + 1;
  for (const auto &pair: most_likely_left_to_right_pair) {
    const auto right_id = pair.second.right_id;
    const auto left_id = pair.first;
    if (proc_right.find(right_id) != std::end(proc_right)) {
      std::cout << "RIGHT TRACK ALREADY PAIRED " << left_id << ", " << right_id << std::endl;
      continue;
    }

    proc_left.emplace(left_id);
    proc_right.emplace(right_id);

    auto left_track = m_tracks_with_3d_left[left_id]->clone();
    auto right_track = m_right_tracks_memo[right_id]->clone();

    // Update left right pairing in case the pairing id is different.
    // Otherwise, keep the track id
    if (left_id != right_id) {
      if (m_verbose)
        std::cout << "PAIRING " << left_id << ", " << right_id << " TO " << last_track_id << std::endl;
      left_track->set_id(last_track_id);
      right_track->set_id(last_track_id);
      last_track_id++;
    } else {
      if (m_verbose)
        std::cout << "KEEPING PAIRING " << left_id << ", " << right_id << std::endl;
    }
    left_tracks.emplace_back(left_track);
    right_tracks.emplace_back(right_track);
  }

  return {proc_left, proc_right};
}

std::tuple<std::set<kwiver::vital::track_id_t>, std::set<kwiver::vital::track_id_t>>
viame::ocv_pair_stereo_tracks::split_paired_tracks_to_new_tracks(
    std::vector<kwiver::vital::track_sptr> &left_tracks, std::vector<kwiver::vital::track_sptr> &right_tracks) {
  const auto ranges = create_split_ranges_from_track_pairs(m_left_to_right_pairing);
  return split_ranges_to_tracks(ranges, left_tracks, right_tracks);
}


void viame::ocv_pair_stereo_tracks::append_paired_frame(const kwiver::vital::track_sptr &left_track,
                                                                  const kwiver::vital::track_sptr &right_track,
                                                                  const kwiver::vital::timestamp &timestamp) {
  // Pair left to right for given frame
  auto pairing = cantor_pairing(left_track->id(), right_track->id());

  if (m_verbose)
    std::cout << "PAIRING (" << pairing << "): " << left_track->id() << ", " << right_track->id() << std::endl;

  if (m_left_to_right_pairing.find(pairing) == std::end(m_left_to_right_pairing))
    m_left_to_right_pairing[pairing] = Pairing{{},
                                               {left_track->id(), right_track->id()}};

  m_left_to_right_pairing[pairing].frame_set.emplace(timestamp.get_frame());
}

void viame::ocv_pair_stereo_tracks::pair_left_right_tracks(
    const std::vector<kwiver::vital::track_sptr> &left_tracks,
    const std::vector<viame::Detections3DPositions> &left_3d_pos,
    const std::vector<kwiver::vital::track_sptr> &right_tracks, const kwiver::vital::timestamp &timestamp) {
  m_detection_pairing->m_verbose = m_verbose;
  m_detection_pairing->m_pairing_method = m_pairing_method;
  m_detection_pairing->m_iou_pair_threshold = m_iou_pair_threshold;

  std::vector<kwiver::vital::track_sptr> filtered_left, filtered_right;
  std::vector<kwiver::vital::detected_object_sptr> filtered_left_detections, filtered_right_detections;
  std::vector<viame::Detections3DPositions> filtered_3d_pos;

  auto get_current_track_detection = [&timestamp](
      const kwiver::vital::track_sptr &track) -> kwiver::vital::detected_object_sptr {
    auto state = std::dynamic_pointer_cast<kwiver::vital::object_track_state>(track->back());
    if (track->last_frame() < timestamp.get_frame() || !state)
      return nullptr;

    return state->detection();
  };

  // Filter tracks which have detections in the current frame only
  for (size_t i_left = 0; i_left < left_tracks.size(); i_left++) {
    const auto &left_track = left_tracks[i_left];
    auto left_detection = get_current_track_detection(left_track);
    if (!left_detection)
      continue;

    filtered_left.emplace_back(left_track);
    filtered_left_detections.emplace_back(left_detection);
    filtered_3d_pos.emplace_back(left_3d_pos[i_left]);
  }

  for (const auto &right_track: right_tracks) {
    auto right_detection = get_current_track_detection(right_track);
    if (!right_detection)
      continue;

    filtered_right.emplace_back(right_track);
    filtered_right_detections.emplace_back(right_detection);
  }

  // Call detection pairing logic
  const auto pairings = m_detection_pairing->pair_left_right_detections(filtered_left_detections, filtered_3d_pos,
                                                                        filtered_right_detections);

  // Append detected frames to memo
  for (const auto &pairing: pairings) {
    append_paired_frame(filtered_left[pairing[0]], filtered_right[pairing[1]], timestamp);
  }
}


std::vector<kwiver::vital::track_sptr> viame::ocv_pair_stereo_tracks::filter_tracks_with_threshold(
    std::vector<kwiver::vital::track_sptr> tracks) const {
  const auto is_outside_detection_number_threshold = [this](const kwiver::vital::track_sptr &track) {
    return ((int) track->size() < m_min_detection_number_threshold) ||
           ((int) track->size() > m_max_detection_number_threshold);
  };

  const auto is_outside_detection_area_threshold = [this](const kwiver::vital::track_sptr &track) {
    double avg_area{};
    for (const auto &state: *track | kwiver::vital::as_object_track) {
      if (state->detection())
        avg_area += state->detection()->bounding_box().area();
    }
    avg_area /= (double) track->size();

    return ((int) avg_area < m_min_detection_surface_threshold_pix) ||
           ((int) avg_area > m_max_detection_surface_threshold_pix);
  };

  tracks.erase(std::remove_if(std::begin(tracks), std::end(tracks), is_outside_detection_number_threshold),
               std::end(tracks));
  tracks.erase(std::remove_if(std::begin(tracks), std::end(tracks), is_outside_detection_area_threshold),
               std::end(tracks));

  return tracks;
}


/// @brief Helper function to erase elements from container given predicate
template<typename ContainerT, typename PredicateT>
void erase_if(ContainerT &items, const PredicateT &predicate) {
  for (auto it = items.begin(); it != items.end();) {
    if (predicate(*it)) it = items.erase(it);
    else ++it;
  }
}

inline std::string to_string(const std::map<size_t, viame::Pairing> &left_to_right_pairing) {
  std::stringstream ss;

  ss << "PAIRINGS TO SPLIT : " << std::endl;
  for (const auto &pair: left_to_right_pairing) {
    std::string print_frames{"{"};
    for (const auto &frame_id: pair.second.frame_set)
      print_frames += "," + std::to_string(frame_id);
    print_frames += "}";

    ss << "ID: " << pair.first << ", left: " << pair.second.left_right_id_pair.left_id << ", right: "
       << pair.second.left_right_id_pair.right_id << ", frames: " << print_frames << std::endl;
  }
  return ss.str();
}

std::vector<viame::ocv_pair_stereo_tracks::Range>
viame::ocv_pair_stereo_tracks::create_split_ranges_from_track_pairs(
    const std::map<size_t, Pairing> &left_to_right_pairing) const {

  if (m_verbose)
    std::cout << to_string(left_to_right_pairing) << std::endl;

  // Find last pairing frame id from all saved pairings
  kwiver::vital::frame_id_t last_pairings_frame_id{};

  for (const auto &pairing: left_to_right_pairing) {
    if (pairing.second.frame_set.empty())
      continue;

    const auto pairing_frame_id = *(pairing.second.frame_set.rbegin());
    if (pairing_frame_id > last_pairings_frame_id)
      last_pairings_frame_id = pairing_frame_id;
  }

  // Init output ID as last unused track ID
  auto last_track_id = last_left_right_track_id() + 1;

  // Initialize ranges being processed, pending ranges, and ranges which have been closed
  std::map<size_t, std::shared_ptr<Range>> open_ranges;
  std::set<std::shared_ptr<Range>> pending_ranges;
  std::vector<Range> ranges;

  // Helper function to find the ranges which are associated with input and need to be removed
  const auto get_pending_ranges_to_remove = [&](const std::shared_ptr<Range> &source_range) {
    std::set<std::shared_ptr<Range >> to_remove;

    if (source_range->detection_count < m_detection_split_threshold)
      return to_remove;

    for (const auto &pending: pending_ranges) {
      if (pending == source_range)
        continue;

      if ((pending->left_id == source_range->left_id) || (pending->right_id == source_range->right_id))
        to_remove.emplace(pending);
    }

    return to_remove;
  };

  // Helper function to remove list of ranges from pending ranges
  const auto remove_pending = [&](const std::shared_ptr<Range> &source_range,
                                  const std::set<std::shared_ptr<Range>> &to_remove) {
    // Remove source range if still pending while detection count is bigger than threshold
    if ((source_range->detection_count >= m_detection_split_threshold) &&
        (pending_ranges.find(source_range) != std::end(pending_ranges)))
      pending_ranges.erase(source_range);

    erase_if(pending_ranges,
             [&](const std::shared_ptr<Range> &range) { return to_remove.find(range) != std::end(to_remove); });
  };

  // Helper function to move input ranges to processed ranges and update open ranges map
  const auto move_open_ranges_to_processed_ranges = [&](const std::set<std::shared_ptr<Range>> &to_remove,
                                                        const std::shared_ptr<Range> &range) {
    auto id_last = range->frame_id_first - 1;


    // Update output ranges
    for (const auto &range_to_remove: to_remove) {
      if (range_to_remove->detection_count < m_detection_split_threshold)
        continue;

      range_to_remove->frame_id_last = id_last;
      ranges.push_back(*range_to_remove);
    }

    // Update open ranges map
    erase_if(open_ranges, [&](const std::pair<size_t, std::shared_ptr<Range>> &pair) {
      return to_remove.find(pair.second) != std::end(to_remove);
    });
  };

  const auto get_overlapping_ranges = [&](const Pairing &pairing) {
    std::vector<std::shared_ptr<Range>> overlapping;
    for (auto &pair: open_ranges) {
      if (pair.second->left_id == pairing.left_right_id_pair.left_id ||
          pair.second->right_id == pairing.left_right_id_pair.right_id)
        overlapping.push_back(pair.second);
    }

    return overlapping;
  };

  const auto create_range_from_pairing = [&](const Pairing &pairing, kwiver::vital::frame_id_t i_frame) {
    auto range = std::make_shared<Range>();
    range->left_id = pairing.left_right_id_pair.left_id;
    range->right_id = pairing.left_right_id_pair.right_id;
    range->new_track_id = last_track_id++;
    range->detection_count = 1;
    range->frame_id_first = i_frame;
    range->frame_id_last = range->frame_id_first + 1;
    return range;
  };

  const auto mark_overlapping_as_pending = [&](const std::vector<std::shared_ptr<Range>> &overlapping) {
    for (const auto &range: overlapping)
      pending_ranges.emplace(range);
  };


  // For each frame
  for (kwiver::vital::frame_id_t i_frame = 0; i_frame <= last_pairings_frame_id; i_frame++) {
    // For each pairing
    for (const auto &pairing: left_to_right_pairing) {
      // If pairing not in current frame -> continue
      if (pairing.second.frame_set.find(i_frame) == std::end(pairing.second.frame_set))
        continue;

      if (open_ranges.find(pairing.first) != std::end(open_ranges)) {
        // If pairing in opened ranges -> Update range and process associated pending ranges
        const auto &range = open_ranges.find(pairing.first)->second;
        range->detection_count += 1;
        range->frame_id_last = i_frame + 1;

        // Remove pending ranges
        const auto to_remove = get_pending_ranges_to_remove(range);
        remove_pending(range, to_remove);
        move_open_ranges_to_processed_ranges(to_remove, range);
      } else {
        // Get overlapping ranges from opened ranges
        const auto overlapping = get_overlapping_ranges(pairing.second);

        // Create new range and mark overlapping as pending
        open_ranges[pairing.first] = create_range_from_pairing(pairing.second, i_frame);
        pending_ranges.emplace(open_ranges[pairing.first]);
        mark_overlapping_as_pending(overlapping);
      }
    }
  }

  for (auto &open_range: open_ranges) {
    // Ignore pending results
    if (open_range.second->detection_count < m_detection_split_threshold)
      continue;

    // Update last frame for all opened ranges to max value
    open_range.second->frame_id_last = std::numeric_limits<int64_t>::max();

    // Add range to ranges to return
    ranges.push_back(*open_range.second);
  }

  // Return consolidated ranges
  return ranges;
}


std::tuple<std::set<kwiver::vital::track_id_t>, std::set<kwiver::vital::track_id_t>>
viame::ocv_pair_stereo_tracks::split_ranges_to_tracks(const std::vector<Range> &ranges,
                                                                std::vector<kwiver::vital::track_sptr> &left_tracks,
                                                                std::vector<kwiver::vital::track_sptr> &right_tracks) {

  const auto split_track = [](const kwiver::vital::track_sptr &track, const Range &range) {
    auto new_track = kwiver::vital::track::create();
    new_track->set_id(range.new_track_id);

    for (const auto &state: *track | kwiver::vital::as_object_track)
      if (state->frame() >= range.frame_id_first && state->frame() < range.frame_id_last)
        new_track->append(state);

    return new_track;
  };

  std::set<kwiver::vital::track_id_t> proc_left, proc_right;
  for (const auto &range: ranges) {
    const auto print_i_last =
        range.frame_id_last == std::numeric_limits<int64_t>::max() ? "i_max" : std::to_string(range.frame_id_last);

    if (m_verbose)
      std::cout << "PAIRING [" << range.frame_id_first << "," << print_i_last << "] " << range.left_id << ", "
                << range.right_id << " TO " << range.new_track_id << std::endl;
    proc_left.emplace(range.left_id);
    proc_right.emplace(range.right_id);

    left_tracks.push_back(split_track(m_tracks_with_3d_left[range.left_id], range));
    right_tracks.push_back(split_track(m_right_tracks_memo[range.right_id], range));
  }

  return std::make_tuple(proc_left, proc_right);
}
