// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file \brief This file contains the implementation for metadata traits.
 */

#include "metadata_traits.h"

namespace kwiver {
namespace vital {

#if 0
template <vital_metadata_tag tag>
struct vital_meta_trait_object
  : public vital_meta_trait_base
{
  virtual std::string name() const override { return vital_meta_trait<tag>::name(); }
  virtual std::string description() const { return vital_meta_trait<tag>::description(); }
  virtual std::type_info const& tag_type() const override { return vital_meta_trait<tag>::tag_type(); }
  virtual bool is_integral() const override { return vital_meta_trait<tag>::is_integral(); }
  virtual bool is_floating_point() const override { return vital_meta_trait<tag>::is_floating_point(); }
  virtual vital_metadata_tag tag() const override { return vital_meta_trait<tag>::tag(); }
};

#endif

#if 01

template <vital_metadata_tag tag>
struct vital_meta_trait_object
  : public vital_meta_trait_base {};

#define DEFINE_VITAL_META_TRAIT(TAG, NAME, T, TD)                       \
  template <>                                                           \
  struct vital_meta_trait_object<VITAL_META_ ## TAG>                    \
    : public vital_meta_trait_base                                      \
  {                                                                     \
    virtual std::string name() const override { return std::string(NAME); } \
    virtual std::string description() const override { return std::string(TD); } \
    virtual std::type_info const& tag_type() const override { return typeid(T); } \
    virtual bool is_integral() const override { return std::is_integral<T>::value; } \
    virtual bool is_floating_point() const override { return std::is_floating_point<T>::value; } \
    virtual vital_metadata_tag tag() const override { return VITAL_META_ ## TAG; } \
    virtual std::unique_ptr<metadata_item> create_metadata_item ( const kwiver::vital::any& data ) const override \
    { \
      return std::unique_ptr<metadata_item>{ \
        new kwiver::vital::typed_metadata< VITAL_META_ ## TAG, T > ( NAME, data ) \
      }; \
    } \
  };

#endif

  KWIVER_VITAL_METADATA_TAGS( DEFINE_VITAL_META_TRAIT )

// ------------------------------------------------------------------
metadata_traits
::metadata_traits()
  : m_logger( kwiver::vital::get_logger( "vital.metadata_traits" ) )
{
  // Create trait table
#define TABLE_ENTRY(TAG, NAME, TYPE, ...)        \
  m_trait_table[VITAL_META_ ## TAG] = trait_ptr( \
    static_cast< vital_meta_trait_base* >(new vital_meta_trait_object<VITAL_META_ ## TAG>() ) );

  KWIVER_VITAL_METADATA_TAGS( TABLE_ENTRY )

#undef TABLE_ENTRY
#undef DEFINE_VITAL_META_TRAIT

}

// ------------------------------------------------------------------
metadata_traits
::~metadata_traits()
{

}

// ------------------------------------------------------------------
vital_meta_trait_base const&
metadata_traits
::find( vital_metadata_tag tag ) const
{
  auto ix = m_trait_table.find( tag );
  if ( ix == m_trait_table.end() )
  {
    LOG_INFO( m_logger, "Could not find trait for tag: " << tag );
    ix = m_trait_table.find(VITAL_META_UNKNOWN);
  }
  return *ix->second;
}

// ------------------------------------------------------------------
std::string
metadata_traits
::tag_to_symbol( vital_metadata_tag tag ) const
{
#define TAG_CASE( TAG, NAME, TYPE, ... ) case VITAL_META_##TAG: return "VITAL_META_" #TAG;

  switch (tag)
  {

    KWIVER_VITAL_METADATA_TAGS( TAG_CASE )

  default:
  case VITAL_META_LAST_TAG:
      return "-- unknown tag code --";
    break;
  } // end switch

#undef TAG_CASE
}

// ------------------------------------------------------------------
std::string
metadata_traits
::tag_to_name( vital_metadata_tag tag ) const
{
  vital_meta_trait_base const& trait = find( tag );
  return trait.name();
}

// ------------------------------------------------------------------
std::string
metadata_traits
::tag_to_description( vital_metadata_tag tag ) const
{
  vital_meta_trait_base const& trait = find( tag );
  return trait.description();
}

} } // end namespace
