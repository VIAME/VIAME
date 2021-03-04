// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief C interface implementation of base algorithm/_def/_impl classes
 */

#include "algorithm.h"

#include <vital/algo/algorithm.h>

#include <vital/bindings/c/helpers/algorithm.h>
#include <vital/bindings/c/helpers/c_utils.h>

// ===========================================================================
// Helper stuff
// ---------------------------------------------------------------------------

namespace kwiver {
namespace vital_c {

/// Global cache for all algorithm instances ever.
SharedPointerCache< kwiver::vital::algorithm,
                    vital_algorithm_t > ALGORITHM_SPTR_CACHE( "algorithm" );

} // end namespace vital_c
} // end namespace kwiver

// ===========================================================================
// Functions on general algorithm pointer
// ---------------------------------------------------------------------------

char const*
vital_algorithm_type_name( vital_algorithm_t *algo,
                           vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "C::algorithm::type_name", eh,
    std::string s( kwiver::vital_c::ALGORITHM_SPTR_CACHE.get( algo )->type_name() );
    return s.c_str();
  );
  return 0;
}

char const*
vital_algorithm_impl_name( vital_algorithm_t const *algo,
                           vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "C::algorithm::impl_name", eh,
    std::string s( kwiver::vital_c::ALGORITHM_SPTR_CACHE.get( algo )->impl_name() );
    return s.c_str();
  );
  return "";
}

/// Get an algorithm implementation's configuration block
vital_config_block_t*
vital_algorithm_get_impl_configuration( vital_algorithm_t *algo,
                                        vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "C::algorithm::get_impl_configuration", eh,
    kwiver::vital::config_block_sptr cb_sptr =
      kwiver::vital_c::ALGORITHM_SPTR_CACHE.get( algo )->get_configuration();
    kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.store( cb_sptr );
    return reinterpret_cast<vital_config_block_t*>( cb_sptr.get() );
  );
  return 0;
}

/// Set this algorithm implementation's properties via a config block
void
vital_algorithm_set_impl_configuration( vital_algorithm_t *algo,
                                        vital_config_block_t *cb,
                                        vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "C::algorithm::set_impl_configuration", eh,
    kwiver::vital_c::ALGORITHM_SPTR_CACHE.get( algo )->set_configuration(
      kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb )
    );
  );
}

/// Check that the algorithm implementation's configuration is valid
bool
vital_algorithm_check_impl_configuration( vital_algorithm_t *algo,
                                          vital_config_block_t *cb,
                                          vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "C::algorithm::check_impl_configuration", eh,
    kwiver::vital_c::ALGORITHM_SPTR_CACHE.get( algo )->check_configuration(
      kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb )
    );
  );
  return false;
}
