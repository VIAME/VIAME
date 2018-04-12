/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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
 * \file
 * \brief Header defining the core interpolate_track_spline algorithm
 */

#ifndef KWIVER_ARROWS_CORE_INTERPOLATE_TRACK_SPLINE_H_
#define KWIVER_ARROWS_CORE_INTERPOLATE_TRACK_SPLINE_H_

#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/algo/interpolate_track.h>


namespace kwiver {
namespace arrows {
namespace core {

/// Fills in missing track segments using spline interpolation
/**
 * This class generates additional track states in between known states using
 * a configurable variety of spline-based interpolation techniques that do not
 * depend on imagery.
 */
class KWIVER_ALGO_CORE_EXPORT interpolate_track_spline
  : public vital::algorithm_impl<interpolate_track_spline,
                                 vital::algo::interpolate_track>
{
public:
  /// Name of the algorithm
  static constexpr char const* name = "spline";

  /// Description of the algorithm
  static constexpr char const* description =
    "Fill in missing object track intervals using spline-based interpolation.";

  /// Default Constructor
  interpolate_track_spline();

  /// Destructor
  virtual ~interpolate_track_spline();

  /// Set this algo's properties via a config block
  virtual void set_configuration(
    vital::config_block_sptr config ) override;

  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration(
    vital::config_block_sptr config ) const override;

  /// Interpolates the states between track states
  virtual kwiver::vital::track_sptr interpolate(
    kwiver::vital::track_sptr init_states ) override;

protected:
  /// private implementation class
  class priv;
  std::unique_ptr<priv> const d_;
};

} } } // end namespace

#endif /* KWIVER_ARROWS_CORE_INTERPOLATE_TRACK_SPLINE_H_ */
