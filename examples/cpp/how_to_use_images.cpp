/*ckwg +29
* Copyright 2017 by Kitware, Inc.
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

#include "arrows/ocv/image_io.h"
#include "arrows/ocv/image_container.h"
#include "arrows/vxl/image_io.h"
#include "arrows/vxl/image_container.h"

#include <opencv2/highgui/highgui.hpp>

#include "vital/types/image.h"
#include "vital/types/image_container.h"
#include "vital/algo/image_io.h"

void how_to_use_images()
{
  // Note that the use of _sptr in objet typing.
  // All vital objects (types, algorithms, etc.) provide a shared_pointer typedef
  // This shared pointer typedef is used through out kwiver to elimate the need of memory ownership managers

  // KWIVER currently can open images using various libraries.
  // The main image libraries used in KWIVER are the OpenCV and VXL libraries
  // All algorithms are implemented/encapsulated in an arrow, and operate on vital classes
  // Image I/O algorithms are derived from the kwiver::vital::image_io algorithm interface
  
  // The following arrows impliment that I/O algorithm interface with the use of OpenCV and VXL
  kwiver::arrows::ocv::image_io ocv_io;
  kwiver::arrows::vxl::image_io vxl_io;

  // The image_io interface is simple, and has a load and save method
  // These methods will operate on the vital object image_container
  // The image_container is intended to be a wrapper for image to facilitate conversion between
  // various representations. It provides limited access to the underlying
  // data and is not intended for direct use in image processing algorithms.
  kwiver::vital::image_container_sptr ocv_img = ocv_io.load("./cat.jpg");
  kwiver::vital::image_container_sptr vxl_img = vxl_io.load("./cat.jpg");

  // Let's use OpenCV to display the images
  cv::Mat mat;
  // First, convert the image to an OpenCV image object
  mat = kwiver::arrows::ocv::image_container::vital_to_ocv(ocv_img->get_image());
  cv::namedWindow("Image loaded by OpenCV", cv::WINDOW_AUTOSIZE);// Create a window for display.
  cv::imshow("Image loaded by OpenCV", mat);                     // Show our image inside it.
  cv::waitKey(0);                                                // Wait for a keystroke in the window

  // We can do the same, even if the image was originally loaded with VXL
  mat = kwiver::arrows::ocv::image_container::vital_to_ocv(vxl_img->get_image());
  cv::namedWindow("Image loaded by VXL", cv::WINDOW_AUTOSIZE);// Create a window for display.
  cv::imshow("Image loaded by VXL", mat);                     // Show our image inside it.
  cv::waitKey(0);                                             // Wait for a keystroke in the window

  // Let's talk about detection sets, tracks, features, and such...
  // Create those objects and generate some new images

  





}