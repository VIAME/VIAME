/*ckwg +29
* Copyright 2017-2018 by Kitware, Inc.
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

#include "vital/algo/image_io.h"
#include "vital/algo/image_filter.h"
#include "vital/algo/split_image.h"

#include "vital/plugin_loader/plugin_manager.h"

// We will be calling some OpenCV code, so we need to include
// some OpenCV related files
#include <opencv2/highgui/highgui.hpp>
#include "arrows/ocv/image_container.h"

#include <kwiversys/SystemTools.hxx>

void how_to_part_01_images()
{
  // Note that the use of _sptr in objet typing.
  // All vital objects (types, algorithms, etc.) provide a shared_pointer typedef
  // This shared pointer typedef is used through out kwiver to elimate the need of memory ownership managers

  // All algorithms are implemented/encapsulated in an arrow, and operate on vital classes
  // There are various algorithms (arrows) that kwiver provides that you can use to analyze imagry
  // In this example, while we will look at a few algorithms, this example highlights the vital data types used by algorithms
  // These vital data types can then be used as inputs or outputs for algorithms.
  // The vital data types are a sort of common 'glue' between dispart algorithms allowing them to work together.

  // Image I/O algorithms are derived from the kwiver::vital::image_io algorithm interface

  // While we could instantiate a particular algorithm object directly with this code
  // kwiver::arrows::ocv::image_io ocv_io;
  // kwiver::arrows::vxl::image_io vxl_io;
  // This would require our application to include specific headers be include in our code
  // and require our application to directly link to OpenCV and cause a dependency

  // A key feature of the KWIVER architecture is the ability to dynamically load available algorithms at runtime.
  // This ability allow you to write your application with a set of basic data types and algorithm interfaces and
  // then dynamically replace or reconfigure algorithms at run time without needing to recompile
  // New algorithms can be dropped on disk at and KWIVER can run them
  // The first thing to do is to tell kwiver to load up all it's plugins (which includes all the algorithms)
  kwiver::vital::plugin_manager::instance().load_all_plugins();

  // Refer to this page : http://kwiver.readthedocs.io/en/latest/vital/images.html
  // Documenting the types and algorithms associated with images:
  //               Various implementations of the algorithm,
  //               The string to use to specify creation of a specific implementation,
  //               The KWIVER CMake option that builds the specific implementation

  ///////////////
  // Image I/O //
  ///////////////

  // The main image libraries used in KWIVER are the OpenCV and VXL libraries
  kwiver::vital::algo::image_io_sptr ocv_io = kwiver::vital::algo::image_io::create("ocv");
  kwiver::vital::algo::image_io_sptr vxl_io = kwiver::vital::algo::image_io::create("vxl");

  // The image_io interface is simple, and has a load and save method
  // These methods will operate on the vital object image_container
  // The image_container is intended to be a wrapper for image to facilitate conversion between
  // various representations. It provides limited access to the underlying
  // data and is not intended for direct use in image processing algorithms.
  kwiver::vital::image_container_sptr ocv_img = ocv_io->load("./cat.jpg");
  kwiver::vital::image_container_sptr vxl_img = vxl_io->load("./cat.jpg");

  // Let's use OpenCV to display the images
  // NOTE, this requires that our application CMakeLists properly find_package(OpenCV)
  // And that we tell our application CMake targets about OpenCV (See the CMakeLists.txt for this file)
  cv::Mat mat;
  // First, convert the image to an OpenCV image object
  mat = kwiver::arrows::ocv::image_container::vital_to_ocv(ocv_img->get_image(), kwiver::arrows::ocv::image_container::RGB_COLOR );
  cv::namedWindow("Image loaded by OpenCV", cv::WINDOW_AUTOSIZE);// Create a window for display.
  cv::imshow("Image loaded by OpenCV", mat);                     // Show our image inside it.
  cv::waitKey(5);
  kwiversys::SystemTools::Delay(2000);                                                   // Wait for 2s
  cvDestroyWindow("Image loaded by OpenCV");

  // We can do the same, even if the image was originally loaded with VXL
  mat = kwiver::arrows::ocv::image_container::vital_to_ocv(vxl_img->get_image(), kwiver::arrows::ocv::image_container::RGB_COLOR);
  cv::namedWindow("Image loaded by VXL", cv::WINDOW_AUTOSIZE);// Create a window for display.
  cv::imshow("Image loaded by VXL", mat);                     // Show our image inside it.
  cv::waitKey(5);
  kwiversys::SystemTools::Delay(2000);                                                // Wait for 2s
  cvDestroyWindow("Image loaded by VXL");

  //////////////////
  // Image Filter //
  //////////////////

  // Currently, there is no arrow implementing image filtering
  //kwiver::vital::algo::image_filter_sptr _filter = kwiver::vital::algo::image_filter::create("<impl_name>");

  /////////////////
  // Split Image //
  /////////////////

  // These algorithms split an image in half (left and right)
  kwiver::vital::algo::split_image_sptr ocv_split = kwiver::vital::algo::split_image::create("ocv");
  kwiver::vital::algo::split_image_sptr vxl_split = kwiver::vital::algo::split_image::create("vxl");

  std::vector<kwiver::vital::image_container_sptr> ocv_imgs = ocv_split->split(vxl_img);
  for (kwiver::vital::image_container_sptr i : ocv_imgs)
  {
    mat = kwiver::arrows::ocv::image_container::vital_to_ocv(i->get_image(), kwiver::arrows::ocv::image_container::RGB_COLOR);
    cv::namedWindow("OpenCV Split Image", cv::WINDOW_AUTOSIZE);// Create a window for display.
    cv::imshow("OpenCV Split Image", mat);                     // Show our image inside it.
    cv::waitKey(5);
    kwiversys::SystemTools::Delay(2000);                                               // Wait for 2s
    cvDestroyWindow("OpenCV Split Image");
  }

  std::vector<kwiver::vital::image_container_sptr> vxl_imgs = ocv_split->split(ocv_img);
  for (kwiver::vital::image_container_sptr i : vxl_imgs)
  {
    mat = kwiver::arrows::ocv::image_container::vital_to_ocv(i->get_image(), kwiver::arrows::ocv::image_container::RGB_COLOR);
    cv::namedWindow("VXL Split Image", cv::WINDOW_AUTOSIZE);// Create a window for display.
    cv::imshow("VXL Split Image", mat);                     // Show our image inside it.
    cv::waitKey(5);
    kwiversys::SystemTools::Delay(2000);                                            // Wait for 2s
    cvDestroyWindow("VXL Split Image");
  }

}
