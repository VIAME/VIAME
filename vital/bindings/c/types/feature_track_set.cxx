// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief C Interface to vital::feature_track implementation
 */

#include "feature_track_set.h"

#include <vital/types/feature_track_set.h>

#include <vital/bindings/c/helpers/c_utils.h>
#include <vital/bindings/c/helpers/descriptor.h>
#include <vital/bindings/c/helpers/feature.h>
#include <vital/bindings/c/helpers/track.h>

using namespace kwiver;

////////////////////////////////////////////////////////////////////////////////
// Track State

/// Create a new track state
vital_track_state_t*
vital_feature_track_state_new( int64_t frame,
                               vital_feature_t *f,
                               vital_descriptor_t *d,
                               vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_feature_track_state_new", eh,
    vital::feature_sptr f_sptr;
    vital::descriptor_sptr d_sptr;
    if( f ) f_sptr = vital_c::FEATURE_SPTR_CACHE.get( f );
    if( d ) d_sptr = vital_c::DESCRIPTOR_SPTR_CACHE.get( d );
    vital::track_state_sptr td_sptr(
      new vital::feature_track_state( frame, f_sptr, d_sptr ) );
    vital_c::TRACK_STATE_SPTR_CACHE.store( td_sptr );
    return reinterpret_cast<vital_track_state_t*>( td_sptr.get() );
  );
  return 0;
}

/// Get a track state's feature
vital_feature_t*
vital_feature_track_state_feature( vital_track_state_t *td,
                                   vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_feature_track_state_feature", eh,
    REINTERP_TYPE( vital::feature_track_state, td, td_ptr );
    // increase cross-boundary reference count if non-null
    if( td_ptr->feature )
    {
      vital_c::FEATURE_SPTR_CACHE.store( td_ptr->feature );
    }
    return reinterpret_cast< vital_feature_t* >( td_ptr->feature.get() );
  );
  return 0;
}

/// Get a track state's descriptor
vital_descriptor_t*
vital_feature_track_state_descriptor( vital_track_state_t *td,
                                      vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_feature_track_state_descriptor", eh,
    REINTERP_TYPE( vital::feature_track_state, td, td_ptr );
    // increase cross-boundary reference count if non-null
    if( td_ptr->descriptor )
    {
      vital_c::DESCRIPTOR_SPTR_CACHE.store( td_ptr->descriptor );
    }
    return reinterpret_cast< vital_descriptor_t* >( td_ptr->descriptor.get() );
  );
  return 0;
}

