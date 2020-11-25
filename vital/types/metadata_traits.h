// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file \brief This file contains the interface for metadata traits.
 */

#ifndef KWIVER_VITAL_METADATA_TRAITS_H_
#define KWIVER_VITAL_METADATA_TRAITS_H_

#include <vital/vital_export.h>

#include <vital/types/geo_point.h>
#include <vital/types/geo_polygon.h>
#include <vital/types/matrix.h>
#include <vital/types/metadata.h>

#include <vital/logger/logger.h>

#include <type_traits>
#include <memory>

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
/// Interface to run-time metadata traits.
/**
 *
 */
struct vital_meta_trait_base
{
  virtual ~vital_meta_trait_base() = default;
  virtual std::string name() const = 0;
  virtual std::string description() const = 0;
  virtual std::type_info const& tag_type() const = 0;
  virtual bool is_integral() const = 0;
  virtual bool is_signed() const = 0;
  virtual bool is_floating_point() const = 0;
  virtual vital_metadata_tag tag() const = 0;
  virtual std::unique_ptr<metadata_item> create_metadata_item( const kwiver::vital::any& data ) const = 0;
};

// ------------------------------------------------------------------
// vital meta compile time traits
//
//

// Macro to define basic metadata trait
// This macro is available for others to create separate sets of traits.
#define DEFINE_VITAL_METADATA_TRAIT(TAG, NAME, T, LD)                   \
  template <>                                                           \
  struct vital_meta_trait<TAG>                                          \
  {                                                                     \
    static std::string name() { return std::string(NAME); }             \
    static std::string  description() { return std::string(LD); }       \
    static std::type_info const& tag_type() { return typeid(T); }       \
    static bool is_integral() { return std::is_integral<T>::value; }    \
    static bool is_signed() { return std::is_signed<T>::value; }    \
    static bool is_floating_point() { return std::is_floating_point<T>::value; } \
    static vital_metadata_tag tag() { return TAG; }                     \
    typedef T type;                                                     \
  };

// Macro to define build-in traits.
#define DEFINE_VITAL_META_TRAIT(TAG, NAME, T, LD)   DEFINE_VITAL_METADATA_TRAIT( VITAL_META_ ## TAG, NAME, T, LD )

//
// Define all compile time metadata traits
//
  KWIVER_VITAL_METADATA_TAGS( DEFINE_VITAL_META_TRAIT )

#undef DEFINE_VITAL_META_TRAIT

// -----------------------------------------------------------------
/**
 *
 *
 */
class VITAL_EXPORT metadata_traits
{
public:
  metadata_traits();
  ~metadata_traits();

  /// Find traits entry for specified tag.
  /**
   * This method returns the metadata trait entry for the
   * specified tag. A default entry is returned if an invalid tag value is specified.
   *
   * @param tag Metadata tag value.
   *
   * @return Metadata traits entry.
   */
  vital_meta_trait_base const& find( vital_metadata_tag tag ) const;

  /// Convert tag value to enum symbol
  /**
   * This method returns the symbol name for the supplied tag.
   *
   * @param tag Metadata tag value
   *
   * @return String representation of the tag symbol.
   */
  std::string tag_to_symbol( vital_metadata_tag tag ) const;

  /// Get name for metadata tag.
  /**
   * This method returns the long form name for the specified tag.
   *
   * @param tag Metadata tag value.
   *
   * @return Long name for this tag.
   */
  std::string tag_to_name( vital_metadata_tag tag ) const;

  // Get metadata tag description
  /**
   * This method returns the long description string for the specified
   * tag.
   *
   * @param tag Metadata tag value.
   *
   * @return Long description for this tag.
   */
  std::string tag_to_description( vital_metadata_tag tag ) const;

private:
  kwiver::vital::logger_handle_t m_logger;

#ifdef VITAL_STD_MAP_UNIQUE_PTR_ALLOWED
  typedef std::unique_ptr< vital_meta_trait_base > trait_ptr;
#else
  typedef std::shared_ptr< vital_meta_trait_base > trait_ptr;
#endif
  std::map< kwiver::vital::vital_metadata_tag, trait_ptr> m_trait_table;

}; // end class metadata_traits

} } // end namespace

#endif
