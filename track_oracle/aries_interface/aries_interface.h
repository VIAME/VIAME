// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_ARIES_INTERFACE_H
#define INCL_ARIES_INTERFACE_H

/*
  This class acts as a weakly-coupled interface to those parts of
  VIRAT which are outside Kitware's source tree, i.e. Lockheed or MCI
  code.

  In particular, we need the mapping of activity strings to their
  index within the classifier vector.  This is maintained by MCI, but
  we do not want to create a hard dependency on having an active ARIES
  build in order to build vidtk.  On the other hand, we want to be able
  to easily (maybe automatically) keep this up-to-date, perhaps by
  automatically generating the list from an up-to-date ARIES checkout
  (much easier to acquire than a build.)  This updating infrastructure
  is deferred until later; for now, we just need to get things going.

  Update 17jul2013 for cross-project activity recognition: add the
  kwe_index_to_activity() routine to kwe activity indices to VIRAT strings.

 */

#include <vital/vital_config.h>
#include <track_oracle/aries_interface/scoring_aries_interface_export.h>

#include <string>
#include <map>
#include <exception>

#undef VIBRANT_AVAILABLE
#ifdef VIBRANT_AVAILABLE
#include <event_detectors/event_types.h>
#endif

namespace kwiver {
namespace track_oracle {

struct aries_interface_impl;

class SCORING_ARIES_INTERFACE_EXPORT
aries_interface_exception: public std::exception
{
  std::string msg;
public:
  explicit aries_interface_exception( const std::string& s )
    : msg( "aries_inteface_typo: couldn't lookup '" + s + "'" ) {}
  virtual ~aries_interface_exception() throw() {}
  virtual const char* what() const throw();
};

class SCORING_ARIES_INTERFACE_EXPORT
aries_interface
{
public:

  // Now string->index operations no longer return maps; see implementation
  // comments for why.

  // Map activity strings to their indices, and vice versa.
  // Return const ref to map, which precludes using operator[],
  // but communicates read-only semantics to the user.

  static size_t activity_to_index( const std::string& s );
  static const std::map< size_t, std::string >& index_to_activity_map();

#ifdef VIBRANT_AVAILABLE
  // kwe index to VIRAT strings; empty if no match

  static std::string kwe_index_to_activity( vidtk::event_types::enum_types kwe_index );
#endif

  // vpd index to VIRAT strings; empty if no match

  static std::string vpd_index_to_activity( unsigned vpd_index );

  // Map activity strings to their PVO associated strings. Due to the
  // information loss in this conversion, there is no vise-versa conversion.
  // Returns const map just like above
  static std::string activity_to_PVO( const std::string& s );
  static const std::map< size_t, std::string >& index_to_PVO_map();

  // Some, but not all, of the activity probabilities should be copied into
  // the PERSON_MOVING and VEHICLE_MOVING slots.

  static bool promote_to_PERSON_MOVING( size_t index );
  static bool promote_to_VEHICLE_MOVING( size_t index );

  // static PVO type values
  static const std::string PVO_PERSON;
  static const std::string PVO_VEHICLE;
  static const std::string PVO_OTHER;
  static const std::string PVO_NULL;

private:
  static aries_interface_impl* p;
};

} //...track_oracle
} //...kwiver

#endif
