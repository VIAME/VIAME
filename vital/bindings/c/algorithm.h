// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief C interface to base algorithm/_def/_impl classes
 */

#ifndef VITAL_C_ALGORITHM_H_
#define VITAL_C_ALGORITHM_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>

#include <vital/bindings/c/common.h>
#include <vital/bindings/c/vital_c_export.h>
#include <vital/bindings/c/config_block.h>
#include <vital/bindings/c/error_handle.h>
#include <vital/bindings/c/types/image_container.h>

/// Opaque pointer to a VITAL Algorithm instance
typedef struct vital_algorithm_s vital_algorithm_t;

// ===========================================================================
// Functions on general algorithm pointer
// ---------------------------------------------------------------------------

/// Return the name of this algorithm
/**
 * \param algo Opaque pointer to algorithm instance.
 * \return String name of the algorithm type.
 */
VITAL_C_EXPORT
char const* vital_algorithm_type_name( vital_algorithm_t *algo,
                                       vital_error_handle_t *eh );

// Return the name of this implementation
/**
 * \param algo Opaque pointer to algorithm instance.
 * \return String name of the algorithm implementation type.
 */
VITAL_C_EXPORT
char const*
vital_algorithm_impl_name( vital_algorithm_t const *algo,
                           vital_error_handle_t *eh );

/// Get an algorithm implementation's configuration block
VITAL_C_EXPORT
vital_config_block_t*
vital_algorithm_get_impl_configuration( vital_algorithm_t *algo,
                                        vital_error_handle_t *eh );

/// Set this algorithm implementation's properties via a config block
VITAL_C_EXPORT
void
vital_algorithm_set_impl_configuration( vital_algorithm_t *algo,
                                        vital_config_block_t *cb,
                                        vital_error_handle_t *eh );

/// Check that the algorithm implementation's configuration is valid
VITAL_C_EXPORT
bool
vital_algorithm_check_impl_configuration( vital_algorithm_t *algo,
                                          vital_config_block_t *cb,
                                          vital_error_handle_t *eh );

/// Common methods for classes that descend from algorithm_def
/**
 * Since the underlying structures in the C++ library use generics at the
 * algorithm_def level, there are a few static and member functions that become
 * specific to the particular algorithm type, requiring there to be multiple
 * versions of the base functions for each type.
 *
 * NOTE: While algorithm destruction is a common method to all algorithms, it
 * is included in the typed interface for implementation reasons.
 */
#define DECLARE_COMMON_ALGO_API( type )                                         \
  /* ==================================================================== */    \
  /* Functions on types (static methods)                                  */    \
  /* -------------------------------------------------------------------- */    \
                                                                                \
  /** Create new instance of a specific algorithm implementation.
   * Returns NULL if there is no implementation currently associated with the
   * name.
   */                                                                           \
  VITAL_C_EXPORT                                                                \
  vital_algorithm_t*                                                            \
  vital_algorithm_##type##_create( char const *impl_name );                     \
                                                                                \
  /** Destroy an algorithm instance of this type */                             \
  VITAL_C_EXPORT                                                                \
  void                                                                          \
  vital_algorithm_##type##_destroy( vital_algorithm_t *algo,                    \
                                    vital_error_handle_t *eh );                 \
                                                                                \
  /** Get a list of registered implementation names for the given type */       \
  VITAL_C_EXPORT                                                                \
  void                                                                          \
  vital_algorithm_##type##_registered_names( unsigned int *length,              \
                                             char ***names );                   \
                                                                                \
  /** Get the configuration for a named algorithm in the given config */        \
  /**
   * NULL may be given for \p algo, which will return a generic
   * configuration for this algorithm type.
   */                                                                           \
  VITAL_C_EXPORT                                                                \
  void                                                                          \
  vital_algorithm_##type##_get_type_config( char const *name,                   \
                                            vital_algorithm_t const *algo,      \
                                            vital_config_block_t *cb,           \
                                            vital_error_handle_t *eh );         \
                                                                                \
  /** Set algorithm properties based on a named configuration in the config */  \
  /**
   * This creates a new vital_algorithm_t instance if the given config block
   * \p cb has a type field for the given \p name and the type is valid, else
   * the \p algo doesn't change (e.g. will remain a NULL pointer of that was
   * what was passed).
   *
   * If given algorithm pointer is changed due to reconstruction, the
   * original pointer is destroyed.
   */                                                                           \
  VITAL_C_EXPORT                                                                \
  void                                                                          \
  vital_algorithm_##type##_set_type_config( char const *name,                   \
                                            vital_config_block_t const *cb,     \
                                            vital_algorithm_t **algo,           \
                                            vital_error_handle_t *eh );         \
                                                                                \
  /** Check the configuration with respect to this algorithm type */            \
  VITAL_C_EXPORT                                                                \
  bool                                                                          \
  vital_algorithm_##type##_check_type_config( char const *name,                 \
                                              vital_config_block_t const *cb,   \
                                              vital_error_handle_t *eh );

#ifdef __cplusplus
}
#endif

#endif //VITAL_C_ALGORITHM_H_
