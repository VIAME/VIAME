/*ckwg +29
 * Copyright 2020 by Kitware, Inc.
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

#ifndef KWIVER_ARROWS_MVG_APPLETS_INIT_CAMERAS_LANDMARKS_H
#define KWIVER_ARROWS_MVG_APPLETS_INIT_CAMERAS_LANDMARKS_H

#include <vital/applets/kwiver_applet.h>
#include <arrows/mvg/applets/kwiver_algo_mvg_applets_export.h>

namespace kwiver {
namespace arrows {
namespace mvg {

class KWIVER_ALGO_MVG_APPLETS_EXPORT init_cameras_landmarks
  : public kwiver::tools::kwiver_applet
{
public:
  init_cameras_landmarks();
  virtual ~init_cameras_landmarks();

  PLUGIN_INFO( "init_cameras_landmarks",
               "Estimate cameras and landmarks from a set of feature tracks");

  virtual int run() override;
  virtual void add_command_options() override;

private:
  class priv;
  std::unique_ptr<priv> d;


}; // end of class

} } } // end namespace

#endif
