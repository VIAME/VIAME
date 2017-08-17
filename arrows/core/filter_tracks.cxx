/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
 * \brief Implementation of core filter_tracks algorithm
 */
#include <arrows/core/filter_tracks.h>
#include <arrows/core/match_matrix.h>

#include <vital/vital_foreach.h>
#include <vital/logger/logger.h>

#include <algorithm>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace core {


/// Private implementation class
class filter_tracks::priv
{
public:
  /// Constructor
  priv()
    : min_track_length(3),
      min_mm_importance(1.0),
      m_logger( vital::get_logger( "arrows.core.filter_tracks" ))
  {
  }

  unsigned int min_track_length;
  double min_mm_importance;
  vital::logger_handle_t m_logger;
};


/// Constructor
filter_tracks
::filter_tracks()
: d_(new priv)
{
}


/// Destructor
filter_tracks
::~filter_tracks()
{
}


/// Get this algorithm's \link vital::config_block configuration block \endlink
  vital::config_block_sptr
filter_tracks
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config =
      vital::algo::filter_tracks::get_configuration();

  config->set_value("min_track_length", d_->min_track_length,
                    "Filter the tracks keeping those covering "
                    "at least this many frames. Set to 0 to disable.");

  config->set_value("min_mm_importance", d_->min_mm_importance,
                    "Filter the tracks with match matrix importance score "
                    "below this threshold. Set to 0 to disable.");

  return config;
}


/// Set this algorithm's properties via a config block
void
filter_tracks
::set_configuration(vital::config_block_sptr config)
{
  d_->min_track_length = config->get_value<unsigned int>("min_track_length",
                                                         d_->min_track_length);
  d_->min_mm_importance = config->get_value<double>("min_mm_importance",
                                                    d_->min_mm_importance);
}


/// Check that the algorithm's configuration vital::config_block is valid
bool
filter_tracks
::check_configuration(vital::config_block_sptr config) const
{
  double min_mm_importance = config->get_value<double>("min_mm_importance",
                                                       d_->min_mm_importance);
  if( min_mm_importance < 0.0 )
  {
    LOG_ERROR( d_->m_logger,
               "min_mm_importance parameter is " << min_mm_importance
               << ", must be non-negative.");
    return false;
  }
  return true;
}


/// Filter feature set
vital::track_set_sptr
filter_tracks
::filter(vital::track_set_sptr tracks) const
{
  if( d_->min_track_length > 1 )
  {
    std::vector<kwiver::vital::track_sptr> trks = tracks->tracks();
    std::vector<kwiver::vital::track_sptr> good_trks;
    VITAL_FOREACH(kwiver::vital::track_sptr t, trks)
    {
      if( t->size() >= d_->min_track_length )
      {
        good_trks.push_back(t);
      }
    }
    tracks = std::make_shared<kwiver::vital::track_set>(good_trks);
  }

  if( d_->min_mm_importance > 0 )
  {
    // compute the match matrix
    std::vector<vital::frame_id_t> frames;
    Eigen::SparseMatrix<unsigned int> mm = kwiver::arrows::match_matrix(tracks, frames);

    // compute the importance scores on the tracks
    std::map<vital::track_id_t, double> importance =
      kwiver::arrows::match_matrix_track_importance(tracks, frames, mm);

    std::vector<kwiver::vital::track_sptr> trks = tracks->tracks();
    std::vector<vital::track_sptr> good_trks;
    VITAL_FOREACH(kwiver::vital::track_sptr t, trks)
    {
      std::map<vital::track_id_t, double>::const_iterator itr;
      if( (itr = importance.find(t->id())) != importance.end() &&
          itr->second >= d_->min_mm_importance)
      {
        good_trks.push_back(t);
      }
    }

    tracks = std::make_shared<kwiver::vital::track_set>(good_trks);
  }
  return tracks;
}


} // end namespace core
} // end namespace arrows
} // end namespace kwiver
