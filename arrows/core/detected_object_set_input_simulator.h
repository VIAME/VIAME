/*ckwg +29
 * Copyright 2018, 2020 by Kitware, Inc.
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
 * \brief Interface for detected_object_set_input_simulator
 */

#ifndef KWIVER_ARROWS_CORE_DETECTED_OBJECT_SET_INPUT_SIMULATOR_H
#define KWIVER_ARROWS_CORE_DETECTED_OBJECT_SET_INPUT_SIMULATOR_H

#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/algo/detected_object_set_input.h>

namespace kwiver {
namespace arrows {
namespace core {

class KWIVER_ALGO_CORE_EXPORT detected_object_set_input_simulator
  : public vital::algo::detected_object_set_input
{
public:
  // NOTE: Keep description in sync with detected_object_set_output_simulator
  PLUGIN_INFO( "simulator",
               "Detected object set reader using SIMULATOR format.\n\n"
               "Detection are generated algorithmicly." )

  detected_object_set_input_simulator();
  virtual ~detected_object_set_input_simulator();

  virtual vital::config_block_sptr get_configuration() const;
  virtual void set_configuration(vital::config_block_sptr config);
  virtual bool check_configuration(vital::config_block_sptr config) const;

  virtual void open( std::string const& filename );
  virtual bool read_set( kwiver::vital::detected_object_set_sptr & set, std::string& image_name );

private:
  class priv;
  std::unique_ptr< priv > d;
};

} } } // end namespace

#endif // KWIVER_ARROWS_CORE_DETECTED_OBJECT_SET_INPUT_SIMULATOR_H
