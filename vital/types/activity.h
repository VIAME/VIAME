// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header for an activity
 */

#ifndef VITAL_TYPES_ACTIVITY_H_
#define VITAL_TYPES_ACTIVITY_H_

#include <vital/types/activity_type.h>
#include <vital/types/object_track_set.h>
#include <vital/types/timestamp.h>

#include <vital/vital_export.h>
#include <vital/vital_types.h>

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
   *        confidence, classifications, starting frame, ending frame and
   *        participants
   *
   * @param id A numeric identifier associated with the activity
   * @param label A label associated with the activity (default=UNDEFINED_ACTIVITY)
   * @param confidence Confidence in the activity (default=-1.0)
   * @param classifications Optional activity classifications
   * @param start Optional timestamp for starting frame of an activity
   * @param end Optional timstamp for ending frame of an activity
   * @param participants Optional participants in the activity
   */
  activity( activity_id_t id,
            activity_label_t label=UNDEFINED_ACTIVITY,
            double confidence=-1.0,
            activity_type_sptr classifications=nullptr,
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
   * @param id  An activity id for the activity
   */
  void set_id( activity_id_t const id );

  /**
   * @brief Get activity label
   *
   * @return Activity label for the activity
   */
  activity_label_t label() const;

  /**
   * @brief Set activity label for the activity
   *
   * @param label  An activity id for the activity
   */
  void set_label( activity_label_t const label );

  /**
   * @brief Get activity type
   *
   * @return the activity type for the activity
   */
  activity_type_sptr type() const;

  /**
   * @brief Set activity type for the activity
   *
   * @param c New classifications for this activity
   */
  void set_type( activity_type_sptr c );

  /**
   * @brief Get activity confidence
   *
   * @return the activity confidence for the activity
   */
  double confidence() const;

  /**
   * @brief Set activity confidence for the activity
   *
   * @param confidence The confidence associated with the activity
   */
  void set_confidence( double confidence );

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
  activity_id_t m_id;
  activity_label_t m_label;
  activity_type_sptr m_type;
  double m_confidence;
  kwiver::vital::object_track_set_sptr m_participants;
  kwiver::vital::timestamp m_start_frame, m_end_frame;
};

} }

#endif
