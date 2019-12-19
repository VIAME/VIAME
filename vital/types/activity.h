/*ckwg +30
 * Copyright 2020 by Kitware, Inc.
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
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 */

/**
 * \file
 * \brief Header for an activity
 */

#ifndef VITAL_TYPES_ACTIVITY_H_
#define VITAL_TYPES_ACTIVITY_H_

#include <vital/vital_export.h>

#include <vital/vital_types.h>
#include <vital/types/activity_type.h>
#include <vital/types/timestamp.h>
#include <vital/types/object_track_set.h>

namespace kwiver {
namespace vital {
// ----------------------------------------------------------------------------
/**
 * @brief Activity class.
 *
 * This class represents an activity.
 */
class VITAL_EXPORT activity
{
public:
  /**
   * @brief Create an empty activity
   *
   * An activity without an activity id, label, confidence or participants
   */
  activity();

  /**
   * @brief Create an activity with activity id with optional activity label,
   *        activity_confidence, activity_type, starting frame, ending frame and
   *        participants
   *
   * @param activity_id A numeric identifier associated with the activity
   * @param activity_label A label associated with the activity (default=UNDEFINED_ACTIVITY)
   * @param activity_confidence Confidence in the activity (default=-1.0)
   * @param activity_type Activity type associated with the activity (default=nullptr)
   * @param start Timestamp for starting frame of an activity (default=-1)
   * @param end Timstamp for ending frame of an activity (default=-1)
   * @param participants Participants in the activity (default=nullptr)
   */
  activity( activity_id_t activity_id,
            activity_label_t activity_label=UNDEFINED_ACTIVITY,
            activity_confidence_t activity_confidence=-1.0,
            activity_type_sptr activity_type=nullptr,
            kwiver::vital::timestamp start=kwiver::vital::timestamp(-1, -1),
            kwiver::vital::timestamp end=kwiver::vital::timestamp(-1, -1),
            kwiver::vital::object_track_set_sptr participants=nullptr );

  /**
   * @brief Get activity id
   *
   * @return Activity id for the activity
   */
  activity_id_t id() const;

  /**
   * @brief Set activity id for the activity
   *
   * @param activity_id  An activity id for the activity
   */
  void set_id( activity_id_t const activity_id );

  /**
   * @brief Get activity label
   *
   * @return Activity label for the activity
   */
  activity_label_t label() const;

  /**
   * @brief Set activity label for the activity
   *
   * @param activity_label  An activity id for the activity
   */
  void set_label( activity_label_t const activity_label );

  /**
   * @brief Get activity type
   *
   * @return the activity type for the activity
   */
  activity_type_sptr activity_type() const;

  /**
   * @brief Set activity type for the activity
   *
   * @param activity_type  An activity type for the activity
   */
  void set_activity_type( activity_type_sptr activity_type );

  /**
   * @brief Get activity confidence
   *
   * @return the activity confidence for the activity
   */
  activity_confidence_t confidence() const;

  /**
   * @brief Set activity confidence for the activity
   *
   * @param activity_confidence The confidence associated with the activity
   */
  void set_confidence( activity_confidence_t activity_confidence );

  /**
   * @brief Get the starting frame for the activity
   *
   * @return Timestamp associated with the starting frame
   */
  kwiver::vital::timestamp start() const;

  /**
   * @brief Set starting frame for the activity
   *
   * @param start frame timestamp for the starting frame
   */
  void set_start( kwiver::vital::timestamp start_frame );

  /**
   * @brief Get the ending frame for the activity
   *
   * @return Timestamp associated with the activity
   */
  kwiver::vital::timestamp end() const;

  /**
   * @brief Set ending frame for the activity
   *
   * @param ending frame timestamp for the activity
   */
  void set_end( kwiver::vital::timestamp end_frame );

  /**
   * @brief Get activity duration
   *
   * @return pair of timestamp representing starting and ending timestamp for activity
   */
  std::pair<kwiver::vital::timestamp, kwiver::vital::timestamp> duration() const;

  /**
   * @brief Get participants
   *
   * @return the participants for the activity
   */
  kwiver::vital::object_track_set_sptr participants() const;

  /**
   * @brief Set participants for the activity
   *
   * @param participants object track set representing trajectories of the participants
   */
  void set_participants( kwiver::vital::object_track_set_sptr participants );

private:
  activity_id_t m_activity_id;
  activity_label_t m_activity_label;
  activity_type_sptr m_activity_type;
  activity_confidence_t m_activity_confidence;
  kwiver::vital::object_track_set_sptr m_participants;
  kwiver::vital::timestamp m_start_frame, m_end_frame;
};

} }

#endif
