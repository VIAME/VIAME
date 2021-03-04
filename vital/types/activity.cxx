// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/activity.h>

#include <utility>

namespace kwiver {
namespace vital {

  activity::activity()
   :m_id{-1},
    m_label{UNDEFINED_ACTIVITY},
    m_type{nullptr},
    m_confidence{-1.0},
    m_participants{nullptr},
    m_start_frame{kwiver::vital::timestamp(-1, -1)},
    m_end_frame{kwiver::vital::timestamp(-1, -1)}
  {}

  activity::activity( activity_id_t id,
                      activity_label_t label,
                      double confidence,
                      activity_type_sptr classifications,
                      kwiver::vital::timestamp start,
                      kwiver::vital::timestamp end,
                      kwiver::vital::object_track_set_sptr participants )
   :m_id{id},
    m_label{label},
    m_type{classifications},
    m_confidence{confidence},
    m_participants{participants},
    m_start_frame{start},
    m_end_frame{end}
  {}

  activity_id_t
  activity::id() const
  {
    return m_id;
  }

  void activity::set_id( activity_id_t const id )
  {
    m_id = id;
  }

  activity_label_t activity::label() const
  {
    return m_label;
  }

  void activity::set_label( activity_label_t const label )
  {
    m_label = label;
  }

  activity_type_sptr activity::type() const
  {
    return m_type;
  }

  void activity::set_type( activity_type_sptr c )
  {
    m_type = c;
  }

  double activity::confidence() const
  {
    return m_confidence;
  }

  void activity::set_confidence( double confidence )
  {
    m_confidence = confidence;
  }

  kwiver::vital::timestamp activity::start() const
  {
    return m_start_frame;
  }

  void activity::set_start( kwiver::vital::timestamp start_frame )
  {
    m_start_frame = start_frame;
  }

  kwiver::vital::timestamp activity::end() const
  {
    return m_end_frame;
  }

  void activity::set_end( kwiver::vital::timestamp end_frame )
  {
    m_end_frame = end_frame;
  }

  std::pair<kwiver::vital::timestamp, kwiver::vital::timestamp> activity::duration() const
  {
    return std::make_pair(m_start_frame, m_end_frame);
  }

  kwiver::vital::object_track_set_sptr activity::participants() const
  {
    return m_participants;
  }

  void activity::set_participants( kwiver::vital::object_track_set_sptr participants )
  {
    m_participants = participants;
  }

} }
