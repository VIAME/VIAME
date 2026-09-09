// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Header file for \link kwiver::vital::algo::query_track_descriptor_set
///        query_track_descriptor_set \endlink

#ifndef VITAL_QUERY_TRACK_DESCRIPTOR_SET_H_
#define VITAL_QUERY_TRACK_DESCRIPTOR_SET_H_

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/algorithm_framework/vital_export.h>

#include <viame/core_types/track.h>
#include <viame/core_types/track_descriptor.h>

namespace kwiver {

namespace vital {

namespace algo {

// ------------------------------------------------------------------
/// Abstract interface for a collection of track descriptors that can be queried
class VITAL_ALGO_EXPORT query_track_descriptor_set
  : public kwiver::vital::algorithm
{
public:
  query_track_descriptor_set();
  PLUGGABLE_INTERFACE( query_track_descriptor_set );

  /// Tuple containing video name, descriptor, and tracks
  typedef std::tuple< std::string,
    vital::track_descriptor_sptr,
    std::vector< vital::track_sptr > > desc_tuple_t;

  /// Set whether object tracks are used for track descriptor history
  ///
  /// \param value If true, the descriptor history is reconstructed from the
  ///              associated object tracks rather than from the descriptors
  ///              alone.
  virtual void use_tracks_for_history( bool value ) = 0;

  /// Look up a track descriptor by its unique identifier
  ///
  /// \param[in]  uid    Unique identifier of the descriptor to retrieve.
  /// \param[out] result Descriptor and its associated tracks. Only written
  ///                    when the lookup succeeds.
  ///
  /// \returns True if a descriptor with \p uid was found, false otherwise.
  virtual bool get_track_descriptor(
    std::string const& uid,
    desc_tuple_t& result ) = 0;
};

/// Shared pointer for base queryable_track_set type
typedef std::shared_ptr< query_track_descriptor_set >
  query_track_descriptor_set_sptr;

} // namespace algo

} // namespace vital

}     // end namespace algo

#endif // VITAL_QUERY_TRACK_DESCRIPTOR_SET_H_
