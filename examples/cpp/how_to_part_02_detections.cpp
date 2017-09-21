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

#include "vital/types/image.h"
#include "vital/types/image_container.h"
#include "vital/types/detected_object.h"
#include "vital/types/detected_object_set.h"

#include "vital/algo/image_io.h"
#include "vital/algo/draw_detected_object_set.h"

#include "vital/plugin_loader/plugin_manager.h"

// We will be calling some OpenCV code, so we need to include
// some OpenCV related files
#include <opencv2/highgui/highgui.hpp>
#include "arrows/ocv/image_container.h"

void how_to_part_02_detections()
{
  // Initialize KWIVER and load up all plugins
  kwiver::vital::plugin_manager::instance().load_all_plugins();

  // Get something to read in an image (see how_to_part_01_images)
  kwiver::vital::algo::image_io_sptr ocv_io = kwiver::vital::algo::image_io::create("ocv");
  kwiver::vital::image_container_sptr ocv_img = ocv_io->load("./cat.jpg");

  // Many vision algorithms are used to detect and identify items in an image.
  // Detectors are any class that implements the kwiver::vital::algo::image_object_detector interface
  // In this example we will explore the detection data types, we are not running any detections,
  // we will only create dummy data in the data types in lieu of running a detection algorithm

  // General detection data is defined by the detected_object class
  // Detectors will take in an image and return a detected_object_set_sptr object
  // A detected_object_set_sptr is comprised of the following data:

  // A bounding box
  // bounding_box_d is a double based box where the top left and bottom right corners are specificed as TODO pixel index?
  // The top left corner is the anchor. A bounding_box_i is interger based to associate corners to pixels in the image
  kwiver::vital::bounding_box_d bbox1(ocv_img->width()*0.25, ocv_img->height()*0.25,
                                      ocv_img->width()*0.75, ocv_img->height()*0.75);
  // Confidence is used for ?? TODO
  double confidence = 1.0;
  // A Classification
  // TODO Is this just a lable?
  kwiver::vital::detected_object_type_sptr type(new kwiver::vital::detected_object_type());
  // TODO Am I doing this right?
  type->set_score("thing 1", 1.0);
  kwiver::vital::detected_object_sptr detection1(new kwiver::vital::detected_object(bbox1,confidence,type));

  // Group multiple detections for an image in a set object
  kwiver::vital::detected_object_set_sptr detections(new kwiver::vital::detected_object_set());
  detections->add(detection1);

  // We can take this detection set and create a new image with the detections overlaid on the image
  // Refer to this page : http://kwiver.readthedocs.io/en/latest/vital/algorithms.html for implementations and build flags
  kwiver::vital::algo::draw_detected_object_set_sptr drawer = kwiver::vital::algo::draw_detected_object_set::create("ocv");
  drawer->set_configuration(drawer->get_configuration());// This will default the configuration 
  kwiver::vital::image_container_sptr img_detections = drawer->draw(detections, ocv_img);

  // Let's see what it looks like
  cv::Mat mat = kwiver::arrows::ocv::image_container::vital_to_ocv(img_detections->get_image());
  cv::namedWindow("Detections", cv::WINDOW_AUTOSIZE);// Create a window for display.
  cv::imshow("Detections", mat);                     // Show our image inside it.
  cv::waitKey(5);
  Sleep(2000);                                       // Wait for 2s
  cvDestroyWindow("Detections");

}