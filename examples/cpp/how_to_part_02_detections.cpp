// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "vital/types/image.h"
#include "vital/types/image_container.h"
#include "vital/types/detected_object.h"
#include "vital/types/detected_object_set.h"

#include "vital/algo/image_io.h"
#include "vital/algo/image_object_detector.h"
#include "vital/algo/draw_detected_object_set.h"
#include "vital/algo/detected_object_set_input.h"
#include "vital/algo/detected_object_set_output.h"

#include "vital/plugin_loader/plugin_manager.h"

// We will be calling some OpenCV code, so we need to include
// some OpenCV related files
#include <opencv2/highgui/highgui.hpp>
#include "arrows/ocv/image_container.h"

#include <kwiversys/SystemTools.hxx>

void how_to_part_02_detections()
{
  // Initialize KWIVER and load up all plugins
  kwiver::vital::plugin_manager::instance().load_all_plugins();

  // Many vision algorithms are used to detect and identify items in an image.
  // Detectors are any class that implements the kwiver::vital::algo::image_object_detector interface
  // In this example we will explore the detection data types.

  // In the following section we will create dummy data in the data types in lieu of running a detection algorithm

  // First, Load an image (see how_to_part_01_images)
  kwiver::vital::algo::image_io_sptr ocv_io = kwiver::vital::algo::image_io::create("ocv");
  kwiver::vital::image_container_sptr ocv_img = ocv_io->load("./soda_circles.jpg");

  // Now let's run a detection algorithm that comes with kwiver
  kwiver::vital::algo::image_object_detector_sptr detector = kwiver::vital::algo::image_object_detector::create("hough_circle");
  kwiver::vital::detected_object_set_sptr hough_detections = detector->detect(ocv_img);

  // We can take this detection set and create a new image with the detections overlaid on the image
  kwiver::vital::algo::draw_detected_object_set_sptr drawer = kwiver::vital::algo::draw_detected_object_set::create("ocv");
  drawer->set_configuration(drawer->get_configuration());// This will default the configuration
  kwiver::vital::image_container_sptr hough_img = drawer->draw(hough_detections, ocv_img);

  // Let's see what it looks like
  cv::Mat hough_mat = kwiver::arrows::ocv::image_container::vital_to_ocv(hough_img->get_image(), kwiver::arrows::ocv::image_container::RGB_COLOR);
  cv::namedWindow("Hough Detections", cv::WINDOW_AUTOSIZE);// Create a window for display.
  cv::imshow("Hough Detections", hough_mat);                     // Show our image inside it.
  cv::waitKey(5);
  kwiversys::SystemTools::Delay(2000);                                       // Wait for 2s
  cv::destroyWindow("Hough Detections");

  // Next, let's look at the detection data structures and we can make them

  // General detection data is defined by the detected_object class
  // Detectors will take in an image and return a detected_object_set_sptr object
  // A detected_object_set_sptr is comprised of the following data:

  // A bounding box
  // bounding_box_d is a double based box where the top left and bottom right corners are specificed as TODO pixel index?
  // The top left corner is the anchor. A bounding_box_i is interger based to associate corners to pixels in the image
  kwiver::vital::bounding_box_d bbox1(ocv_img->width()*0.25, ocv_img->height()*0.25,
                                      ocv_img->width()*0.75, ocv_img->height()*0.75);
  // The confidence value is the confidence associated with the detection.
  // It should be a probability (0..1) that the detector is sure that it has identified what it is supposed to find.
  double confidence1 = 1.0;

  // A Classification
  // The detected_object_type is created by a classifier which is sometimes part of the detector.
  // It is a group of name / value pairs.The name being the name of the class.
  // The score is the probability that the object is that class.
  // It is optional and not required for a detected object although most examples provide one just to be complete.

  kwiver::vital::detected_object_type_sptr type1(new kwiver::vital::detected_object_type());
  // This can have multiple entries / scores
  type1->set_score("car", 0.03);
  type1->set_score("fish", 0.52);
  type1->set_score("flag pole", 0.23);
  // Put it all together to make a detection
  kwiver::vital::detected_object_sptr detection1(new kwiver::vital::detected_object(bbox1, confidence1, type1));
  detection1->set_detector_name("center");

  // Let's add a few more detections to our detection set and write it out in various formats
  kwiver::vital::bounding_box_d bbox2(ocv_img->width()*0.05, ocv_img->height()*0.05,
    ocv_img->width()*0.55, ocv_img->height()*0.55);
  double confidence2 = 0.50;
  kwiver::vital::detected_object_type_sptr type2(new kwiver::vital::detected_object_type());
  type2->set_score("car", 0.04);
  type2->set_score("fish", 0.12);
  type2->set_score("flag pole", 0.67);
  kwiver::vital::detected_object_sptr detection2(new kwiver::vital::detected_object(bbox2, confidence2, type2));
  detection2->set_detector_name("upper left");

  kwiver::vital::bounding_box_d bbox3(ocv_img->width()*0.45, ocv_img->height()*0.45,
    ocv_img->width()*0.95, ocv_img->height()*0.95);
  double confidence3 = 0.75;
  kwiver::vital::detected_object_type_sptr type3(new kwiver::vital::detected_object_type());
  type3->set_score("car", 0.22);
  type3->set_score("fish", 0.08);
  type3->set_score("flag pole", 0.07);
  kwiver::vital::detected_object_sptr detection3(new kwiver::vital::detected_object(bbox3, confidence3, type3));
  detection3->set_detector_name("lower right");

  // Group multiple detections for an image in a set object
  kwiver::vital::detected_object_set_sptr detections(new kwiver::vital::detected_object_set());
  detections->add(detection1);
  detections->add(detection2);
  detections->add(detection3);

  kwiver::vital::image_container_sptr img_detections = drawer->draw(detections, ocv_img);
  // Let's see what it looks like
  cv::Mat mat = kwiver::arrows::ocv::image_container::vital_to_ocv(img_detections->get_image(), kwiver::arrows::ocv::image_container::RGB_COLOR);
  cv::namedWindow("Detections", cv::WINDOW_AUTOSIZE);// Create a window for display.
  cv::imshow("Detections", mat);                     // Show our image inside it.
  cv::waitKey(5);
  kwiversys::SystemTools::Delay(2000);                                       // Wait for 2s
  cv::destroyWindow("Detections");

  kwiver::vital::algo::detected_object_set_output_sptr kpf_writer = kwiver::vital::algo::detected_object_set_output::create("kpf_output");
  kwiver::vital::algo::detected_object_set_input_sptr kpf_reader = kwiver::vital::algo::detected_object_set_input::create("kpf_input");
  if (kpf_writer == nullptr)
  {
    std::cerr << "Make sure you have built the kpf arrow, which requires fletch to have yaml" << std::endl;
  }
  else
  {
    kpf_writer->open("detected_object_set.kpf");
    kpf_writer->write_set(detections, "");

    // Now let's read the kpf data back in
    std::string image_name;
    kwiver::vital::detected_object_set_sptr kpf_detections;
    kpf_reader->open("detected_object_set.kpf");
    kpf_reader->read_set(kpf_detections, image_name);

    auto ie = kpf_detections->cend();
    for (auto det = kpf_detections->cbegin(); det != ie; ++det)
    {
      const kwiver::vital::bounding_box_d bbox((*det)->bounding_box());

      std::stringstream ss;
      ss << "detector_name " << (*det)->detector_name() << "\n"
         << "bounding box :" << "x1(" << bbox.min_x() << ") y1(" << bbox.min_y() << ") x2(" << bbox.max_x() << ") y2(" << bbox.max_y() << ") \n"
         << "confidence : " << (*det)->confidence() << "\n"
         << "classifications : " << "\n";
      for (auto t : *(*det)->type())
        ss << "\t type : " << t.first << " " << t.second;
      std::cout << ss.str();
    }
  }

}
