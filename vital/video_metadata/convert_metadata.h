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
 * \file \brief This file contains the internal interface for
 * converter class.
 */

#ifndef KWIVER_VITAL_CONVERT_METADATA_H
#define KWIVER_VITAL_CONVERT_METADATA_H

#include <vital/video_metadata/vital_video_metadata_export.h>

#include <vital/video_metadata/video_metadata.h>
#include <vital/video_metadata/video_metadata_traits.h>

#include <vital/klv/klv_0601.h>
#include <vital/klv/klv_0104.h>
#include <vital/klv/klv_parse.h>
#include <vital/util/any_converter.h>

#include <vital/logger/logger.h>

namespace kwiver {
namespace vital {

// usage for creating metadata items
#define NEW_METADATA_ITEM( TAG, DATA )                    \
  new typed_metadata< TAG, vital_meta_trait<TAG>::type >  \
  ( vital_meta_trait<TAG>::name(), DATA )

// ----------------------------------------------------------------
/**
 * @brief
 *
 */
class VITAL_VIDEO_METADATA_EXPORT convert_metadata
{
public:
  // -- CONSTRUCTORS --
  convert_metadata();
  ~convert_metadata();

  /**
   * @brief Convert raw metadata packet into vital metadata entries.
   *
   * @param[in] klv Raw metadata packet containing UDS key
   * @param[in,out] metadata Collection of metadata this updated.
   *
   * @throws klv_exception When error encountered.
   */
   void convert( klv_data const& klv, video_metadata& metadata );

private:

  void convert_0601_metadata( klv_lds_vector_t const& lds, video_metadata& metadata );
  void convert_0104_metadata( klv_uds_vector_t const& uds, video_metadata& metadata );

  kwiver::vital::any normalize_0601_tag_data( klv_0601_tag tag,
                                              kwiver::vital::vital_metadata_tag vital_tag,
                                              kwiver::vital::any const& data );

  kwiver::vital::any normalize_0104_tag_data( klv_0104::tag tag,
                                            kwiver::vital::vital_metadata_tag vital_tag,
                                            kwiver::vital::any const& data );

  kwiver::vital::logger_handle_t m_logger;

  any_converter< double > convert_to_double;
  any_converter< uint64_t > convert_to_int;

  video_metadata_traits m_metadata_traits;

}; // end class convert_metadata

} } // end namespace

#endif /* KWIVER_VITAL_CONVERT_METADATA_H */
