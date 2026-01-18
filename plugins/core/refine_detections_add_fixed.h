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

#ifndef VIAME_CORE_REFINE_DETECTIONS_ADD_FIXED_H
#define VIAME_CORE_REFINE_DETECTIONS_ADD_FIXED_H

#include "viame_core_export.h"

#include <vital/algo/refine_detections.h>
#include "viame_algorithm_plugin_interface.h"

namespace viame {

// -----------------------------------------------------------------------------
/**
 * \class refine_detections_add_fixed
 *
 * \brief Prunes overlapping detections
 *
 * \iports
 * \iport{detections}
 *
 * \oports
 * \oport{pruned_detections}
 */
class VIAME_CORE_EXPORT refine_detections_add_fixed
  : public kwiver::vital::algo::refine_detections
{

public:
  VIAME_ALGORITHM_PLUGIN_INTERFACE( refine_detections_add_fixed )

  PLUGIN_INFO( "add_fixed",
    "Adds a fixed detection into the current set.\n\n"
    "The fixed detection can be either a config defined box or "
    "based on the input image size." )

  refine_detections_add_fixed();
  virtual ~refine_detections_add_fixed();

  /// Get this algorithm's \link kwiver::vital::config_block configuration block \endlink
  virtual kwiver::vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration(kwiver::vital::config_block_sptr config);
  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration(kwiver::vital::config_block_sptr config) const;

  /// Refine all object detections on the provided image
  /**
   * This method analyzes the supplied image and and detections on it,
   * returning a refined set of detections.
   *
   * \param image_data the image pixels
   * \param detections detected objects
   * \returns vector of image objects refined
   */
  virtual kwiver::vital::detected_object_set_sptr
  refine( kwiver::vital::image_container_sptr image_data,
          kwiver::vital::detected_object_set_sptr detections ) const;

private:

  /// private implementation class
  class priv;
  const std::unique_ptr<priv> d_;

 }; // end class refine_detections_add_fixed


} // end namespace viame

#endif // VIAME_CORE_REFINE_DETECTIONS_ADD_FIXED_H
