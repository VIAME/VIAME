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
 * \brief Header for refine detections using SVM
 */

#ifndef VIAME_SVM_REFINE_DETECTIONS_SVM_H
#define VIAME_SVM_REFINE_DETECTIONS_SVM_H

#include "viame_svm_export.h"

#include <vital/algo/refine_detections.h>

namespace viame {

namespace kv = kwiver::vital;

/// A class for drawing various information about feature tracks
class VIAME_SVM_EXPORT refine_detections_svm
: public kv::algorithm_impl<refine_detections_svm,
    kv::algo::refine_detections>
{
public:

  PLUGIN_INFO( "svm_refiner",
               "Refine detections using SVM." )

  /// Constructor
  refine_detections_svm();

  /// Destructor
  virtual ~refine_detections_svm();

  /// Get this algorithm's \link kwiver::vital::config_block configuration block \endlink
  virtual kv::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration(kv::config_block_sptr config);
  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration(kv::config_block_sptr config) const;

  /// Refine all object detections on the provided image
  /**
   * This method analyzes the supplied image and and detections on it,
   * returning a refined set of detections.
   *
   * \param image_data the image pixels
   * \param detections detected objects
   * \returns vector of image objects refined
   */
  virtual kv::detected_object_set_sptr
  refine( kv::image_container_sptr image_data,
          kv::detected_object_set_sptr detections ) const;

private:

  /// Private implementation class
  class priv;
  const std::unique_ptr<priv> d_;
};

} // end namespace viame

#endif // VIAME_SVM_REFINE_DETECTIONS_SVM_H
