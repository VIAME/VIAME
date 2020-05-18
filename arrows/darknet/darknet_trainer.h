/*ckwg +29
 * Copyright 2017-2018, 2020 by Kitware, Inc.
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

#ifndef KWIVER_ARROWS_DARKNET_TRAINER
#define KWIVER_ARROWS_DARKNET_TRAINER


#include <arrows/darknet/kwiver_algo_darknet_export.h>

#include <vital/algo/train_detector.h>

namespace kwiver {
namespace arrows {
namespace darknet {

// ----------------------------------------------------------------
/**
 * @brief Darknet Training Utility Class
 */
class KWIVER_ALGO_DARKNET_EXPORT darknet_trainer
  : public vital::algo::train_detector
{
public:
  darknet_trainer();
  virtual ~darknet_trainer();

  PLUGIN_INFO( "darknet",
               "Training utility for darknet." )

  vital::config_block_sptr get_configuration() const override;

  void set_configuration( vital::config_block_sptr config ) override;
  bool check_configuration( vital::config_block_sptr config ) const override;

  void train_from_disk( vital::category_hierarchy_sptr object_labels,
                        std::vector< std::string > train_image_names,
                        std::vector< vital::detected_object_set_sptr > train_groundtruth,
                        std::vector< std::string > test_image_names,
                        std::vector< vital::detected_object_set_sptr > test_groundtruth ) override;

private:
  class priv;

  const std::unique_ptr< priv > d;
};

} } }

#endif /* KWIVER_ARROWS_DARKNET_TRAINER */
