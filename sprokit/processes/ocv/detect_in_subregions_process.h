/*ckwg +29
 * Copyright 2017-2019 by Kitware, Inc.
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

#ifndef _KWIVER_DETECT_IN_SUBREGIONS_PROCESS_H
#define _KWIVER_DETECT_IN_SUBREGIONS_PROCESS_H

#include <sprokit/pipeline/process.h>

#include <vital/config/config_block.h>

#include "kwiver_processes_ocv_export.h"

namespace kwiver
{

// -----------------------------------------------------------------------------
/**
 * @brief Detect in subregions process.
 *
 * This process is intended to be used after an initial object detection step
 * identifies potentially interesting regions within an image, defined by a
 * detected_object_set, to run a subsequent, potentially more
 * expensive, detector/classifier algorithm. Each bounding box from the input
 * detected_object_set defines a sub-image, which is passed to the specified
 * detection algorithm. The result is a new, presumably more accurate,
 * detected_object_set, which may not have direct correlation with the input set
 * of image_object_detection other than being contained within the union of its
 * bounding boxes.
 *
 */
class KWIVER_PROCESSES_OCV_NO_EXPORT detect_in_subregions_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "detect_in_subregions",
               "Run a detection algorithm on all of the chips represented "
               "by an incoming detected_object_set" )

  detect_in_subregions_process( kwiver::vital::config_block_sptr const& config );
  virtual ~detect_in_subregions_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;

}; // end class detect_in_subregions_process



} // end namespace

#endif /* _KWIVER_DETECT_IN_SUBREGIONS_PROCESS_H */
