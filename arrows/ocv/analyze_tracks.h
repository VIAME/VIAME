/*ckwg +29
 * Copyright 2014-2016, 2019-2020 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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
 * \file
 * \brief Header for OpenCV analyze_tracks algorithm
 */

#ifndef KWIVER_ARROWS_OCV_ANALYZE_TRACKS_H_
#define KWIVER_ARROWS_OCV_ANALYZE_TRACKS_H_

#include <arrows/ocv/kwiver_algo_ocv_export.h>

#include <vital/algo/analyze_tracks.h>

namespace kwiver {
namespace arrows {
namespace ocv {

/// A class for outputting various debug info about feature tracks
class KWIVER_ALGO_OCV_EXPORT analyze_tracks
: public vital::algo::analyze_tracks
{
public:
  PLUGIN_INFO( "ocv",
               "Use OpenCV to analyze statistics of feature tracks." )

  /// Constructor
  analyze_tracks();

  /// Destructor
  virtual ~analyze_tracks();

  /// Get this algorithm's \link vital::config_block configuration block \endlink
  virtual vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration(vital::config_block_sptr config);
  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration(vital::config_block_sptr config) const;

  /// Output various information about the tracks stored in the input set.
  /**
   * \param [in] track_set the tracks to analyze
   * \param [in] stream an output stream to write data onto
   */
  virtual void
  print_info(vital::track_set_sptr track_set,
             stream_t& stream = std::cout) const;

private:

  /// private implementation class
  class priv;
  const std::unique_ptr<priv> d_;
};

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver

#endif
