/*ckwg +29
 * Copyright 2016-2018 by Kitware, Inc.
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
 * \brief Interface for detected_object_set_input_kw18
 */

#ifndef KWIVER_ARROWS_CORE_DETECTED_OBJECT_SET_INPUT_KW18_H
#define KWIVER_ARROWS_CORE_DETECTED_OBJECT_SET_INPUT_KW18_H

#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/algo/detected_object_set_input.h>

#include <memory>

namespace kwiver {
namespace arrows {
namespace core {

class KWIVER_ALGO_CORE_EXPORT detected_object_set_input_kw18
  : public vital::algorithm_impl<detected_object_set_input_kw18, vital::algo::detected_object_set_input>
{
public:
  static constexpr char const* name = "kw18";

  // NOTE: Keep description in sync with detected_object_set_output_kw18
  static constexpr char const* description =
    "Detected object set reader using kw18 format.\n\n"
    "  - Column(s) 1: Track-id\n"
    "  - Column(s) 2: Track-length (number of detections)\n"
    "  - Column(s) 3: Frame-number (-1 if not available)\n"
    "  - Column(s) 4-5: Tracking-plane-loc(x,y) (could be same as World-loc)\n"
    "  - Column(s) 6-7: Velocity(x,y)\n"
    "  - Column(s) 8-9: Image-loc(x,y)\n"
    "  - Column(s) 10-13: Img-bbox(TL_x,TL_y,BR_x,BR_y)"
    " (location of top-left & bottom-right vertices)\n"
    "  - Column(s) 14: Area\n"
    "  - Column(s) 15-17: World-loc(x,y,z)"
    " (longitude, latitude, 0 - when available)\n"
    "  - Column(s) 18: Timesetamp (-1 if not available)\n"
    "  - Column(s) 19: Track-confidence (-1 if not available)\n";

  detected_object_set_input_kw18();
  virtual ~detected_object_set_input_kw18();

  virtual void set_configuration(vital::config_block_sptr config);
  virtual bool check_configuration(vital::config_block_sptr config) const;

  virtual bool read_set( kwiver::vital::detected_object_set_sptr & set, std::string& image_name );

private:
  virtual void new_stream();

  class priv;
  std::unique_ptr< priv > d;
};

} } } // end namespace

#endif // KWIVER_ARROWS_CORE_DETECTED_OBJECT_SET_INPUT_KW18_H
