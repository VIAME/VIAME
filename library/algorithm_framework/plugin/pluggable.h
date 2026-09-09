// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_VITAL_PLUGGABLE_H_
#define KWIVER_VITAL_PLUGGABLE_H_

#include <memory>
#include <viame/algorithm_framework/plugin/vital_vpm_export.h>

namespace kwiver::vital {

// ----------------------------------------------------------------------------
// Pluggable base-class

class pluggable;
typedef std::shared_ptr< pluggable > pluggable_sptr;

/**
 * Base-class for pluggable classes.
 *
 * This base provides minimal structure and more acts as a means of
 * categorization so that factories have a basic type to handle.
 *
 * This interface intentionally does not utilize `config_block` in any
 * definitions so that we can have plugins in our configuration world
 * -- NOTE: This could be revised to have the config stuff that are plugins to
 *          not live in the config module if that is feasible (e.g. they're
 *          algorithms in a way).
 */
class VITAL_VPM_EXPORT pluggable
{
public:
  /// Expected static functions:

  /**
   * Provide the human-readable string name of the interface.
   *
   * This is to be defined by the derived classes the define abstract
   * interfaces. Concrete classes may define this, but it makes less sense to do
   */
  // static std::string interface_name() { return "pluggable"; };

  /**
   * Plural for handling rare multiple inheritance scenarios?
   */
  // static std::set<std::string> interface_names();

  /**
   * Curry construction of this concrete class from an input config_block
   * instance.
   * This must be defined on concrete implementations as this is what will
   * return a real instance pointer.
   */
  // static pluggable_sptr from_config( config_block const& cb );

  /// Set into a config-block the default configuration for this concrete type.
  /// Result `cb` state may not be valid for construction, but should at least
  /// provide all the keys required.
  // static void get_default_config( config_block & cb );

  // TODO: Is this even really needed?

  /**
   * @brief Set into the given config_block the configuration of this instance
   *
   * This should provide keys into the `cb` that match the keys provided in the
   * default configuration and be accepted by `from_config`.
   */
//  virtual void get_config( config_block & cb ) const = 0;

  virtual ~pluggable() = default;

protected:
  // Protected constructor to make non-constructable by itself.
  pluggable() = default;
};

// ----------------------------------------------------------------------------
// Static-method Existence Helpers

// Reformatting this seems to take `uncrustify` into an infinite loop of some
// kind (killed after 5 minutes).
// UNCRUST-OFF
#define CREATE_HAS_CHECK( funcname ) \
  template< typename T > \
  class has_##funcname final \
  { \
  private: \
    typedef char r1[1]; \
    typedef char r2[2]; \
    template <typename C> static r1& test( decltype( &C::funcname ) ); \
    template <typename C> static r2& test( ... ); \
  public: \
    enum { value = sizeof(test<T>(nullptr)) == sizeof(r1) }; \
  }
// UNCRUST-ON

/**
 * Use SFINAE To check if the templated type has "interface_name" static method.
 *
 * Usage example:
 * @code
 *   static_assert( has_interface_name<T>::value );
 * @endcode
 */
CREATE_HAS_CHECK( interface_name );

/**
 * Use SFINAE To check if the templated type has "from_config" static method.
 *
 * Usage example:
 * @code
 *   static_assert( has_from_config<T>::value );
 * @endcode
 */
CREATE_HAS_CHECK( from_config );

/**
 * Use SFINAE To check if the templated type has "get_default_config" static
 * method.
 *
 * Usage example:
 * @code
 *   static_assert( has_get_default_config<T>::value );
 * @endcode
 */
CREATE_HAS_CHECK( get_default_config );

// Clean up our macro.
#undef CREATE_HAS_CHECK

} // namespace kwiver::vital

#endif // KWIVER_VITAL_PLUGGABLE_H_
