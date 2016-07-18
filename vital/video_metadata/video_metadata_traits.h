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
 * \file \brief This file contains the interface for video metadata traits.
 */

#ifndef KWIVER_VITAL_VIDEO_METADATA_TRAITS_H
#define KWIVER_VITAL_VIDEO_METADATA_TRAITS_H

#include <vital/video_metadata/video_metadata_tags.h>
#include <vital/video_metadata/video_metadata.h> // needed for corner points

#include <vital/types/geo_lat_lon.h>
#include <vital/types/geo_corner_points.h>

#include <vital/logger/logger.h>

#include <type_traits>
#include <memory>

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
/// Interface to run-time video metadata traits.
/**
 *
 */
struct vital_meta_trait_base
{
  virtual ~vital_meta_trait_base() {}
  virtual std::string name() const = 0;
  virtual std::type_info const& tag_type() const = 0;
  virtual bool is_integral() const = 0;
  virtual bool is_floating_point() const = 0;
  virtual vital_metadata_tag tag() const = 0;
};


// ------------------------------------------------------------------
// vital meta compile time traits
//
//
template <vital_metadata_tag tag> struct vital_meta_trait;

#define DEFINE_VITAL_META_TRAIT(TAG, NAME, T)                           \
  template <>                                                           \
  struct vital_meta_trait<VITAL_META_ ## TAG>                           \
  {                                                                     \
    static std::string name() { return NAME; }                          \
    static std::type_info const& tag_type() { return typeid(T); }       \
    static bool is_integral() { return std::is_integral<T>::value; }    \
    static bool is_floating_point() { return std::is_floating_point<T>::value; } \
    static vital_metadata_tag tag() { return VITAL_META_ ## TAG; }      \
    typedef T type;                                                     \
  };

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
class VITAL_EXPORT video_metadata_traits
{
public:
  video_metadata_traits();
  ~video_metadata_traits();


  /// Find traits entry for specified tag.
  /**
   * This method returns the video metadata trait entry for the
   * specified tag. A default entry is returned if an invalid tag value is specified.
   *
   * @param tag Metadata tag value.
   *
   * @return Video metadata traits entry.
   */
  vital_meta_trait_base const& find( vital_metadata_tag tag ) const;


  /// Get type representation for vital metadata tag. //+ move to convert_metadata
  /**
   * This method returns the type id string for the specified vital
   * metadata tag.
   *
   * @param tag Code for metadata tag.
   *
   * @return Type info for this tag
   */
  std::type_info const& typeid_for_tag( vital_metadata_tag tag ) const;


  /// Convert tag value to enum symbol
  /**
   * This method returns the symbol name for the supplied tag.
   *
   * @param tag Video metadata tag value
   *
   * @return String representation of the tag symbol.
   */
  std::string tag_to_symbol( vital_metadata_tag tag ) const;


  /// Get name for video metadata tag.
  /**
   * This method returns the long form name for the specified tag.
   *
   * @param tag Video mdtadata tag value.
   *
   * @return Long name for this tag.
   */
  std::string tag_to_name( vital_metadata_tag tag ) const;



private:
  kwiver::vital::logger_handle_t m_logger;

#ifdef VITAL_STD_MAP_UNIQUE_PTR_ALLOWED
  typedef std::unique_ptr< vital_meta_trait_base > trait_ptr;
#else
  typedef std::shared_ptr< vital_meta_trait_base > trait_ptr;
#endif
  std::map< kwiver::vital::vital_metadata_tag, trait_ptr> m_trait_table;

}; // end class video_metadata_traits

} } // end namespace

#endif /* KWIVER_VITAL_VIDEO_METADATA_TRAITS_H */
