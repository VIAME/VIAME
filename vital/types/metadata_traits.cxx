/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * \file \brief This file contains the implementation for metadata traits.
 */

#include "video_metadata_traits.h"

namespace kwiver {
namespace vital {

#if 0
template <vital_metadata_tag tag>
struct vital_meta_trait_object
  : public vital_meta_trait_base
{
  virtual ~vital_meta_trait_object() {}
  virtual std::string name() const { return vital_meta_trait<tag>::name(); }
  virtual std::type_info const& tag_type() const { return vital_meta_trait<tag>::tag_type(); }
  virtual bool is_integral() const { return vital_meta_trait<tag>::is_integral(); }
  virtual bool is_floating_point() const { return vital_meta_trait<tag>::is_floating_point(); }
  virtual vital_metadata_tag tag() const { return vital_meta_trait<tag>::tag(); }
};
#endif

#if 01

template <vital_metadata_tag tag>
struct vital_meta_trait_object
  : public vital_meta_trait_base {};

#define DEFINE_VITAL_META_TRAIT(TAG, NAME, T)                           \
  template <>                                                           \
  struct vital_meta_trait_object<VITAL_META_ ## TAG>                    \
    : public vital_meta_trait_base                                      \
  {                                                                     \
    virtual ~vital_meta_trait_object() {}                               \
    virtual std::string name() const { return NAME; }                   \
    virtual std::type_info const& tag_type() const { return typeid(T); } \
    virtual bool is_integral() const { return std::is_integral<T>::value; } \
    virtual bool is_floating_point() const { return std::is_floating_point<T>::value; } \
    virtual vital_metadata_tag tag() const { return VITAL_META_ ## TAG; } \
  };
#endif

  KWIVER_VITAL_METADATA_TAGS( DEFINE_VITAL_META_TRAIT )

// ------------------------------------------------------------------
video_metadata_traits
::video_metadata_traits()
  : m_logger( kwiver::vital::get_logger( "vital.video_metadata_traits" ) )
{
  // Create trait table
#define TABLE_ENTRY(TAG, NAME, TYPE) \
  m_trait_table[VITAL_META_ ## TAG] = trait_ptr( \
    static_cast< vital_meta_trait_base* >(new vital_meta_trait_object<VITAL_META_ ## TAG>() ) );

  KWIVER_VITAL_METADATA_TAGS( TABLE_ENTRY )

#undef TABLE_ENTRY
#undef DEFINE_VITAL_META_TRAIT

}


// ------------------------------------------------------------------
video_metadata_traits
::~video_metadata_traits()
{

}


// ------------------------------------------------------------------
vital_meta_trait_base const&
video_metadata_traits
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
std::type_info const&
video_metadata_traits
::typeid_for_tag( vital_metadata_tag tag ) const
{
  vital_meta_trait_base const& trait = find( tag );
  return trait.tag_type();
}


// ------------------------------------------------------------------
std::string
video_metadata_traits
::tag_to_symbol( vital_metadata_tag tag ) const
{
#define TAG_CASE( TAG, NAME, TYPE ) case VITAL_META_##TAG: return "VITAL_META_" #TAG;

  switch (tag)
  {

    KWIVER_VITAL_METADATA_TAGS( TAG_CASE )

  default:
    return "-- unknown tag code --";
    break;
  } // end switch

#undef TAG_CASE
}


// ------------------------------------------------------------------
std::string
video_metadata_traits
::tag_to_name( vital_metadata_tag tag ) const
{
  vital_meta_trait_base const& trait = find( tag );
  return trait.name();
}


} } // end namespace
