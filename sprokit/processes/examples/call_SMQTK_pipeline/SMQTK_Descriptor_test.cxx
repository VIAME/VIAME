// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "SMQTK_Descriptor.h"
#include <iostream>

// ==================================================================

int main(int argc, char *argv[])
{

  if (argc < 3 )
  {
    std::cerr << "usage: " << argv[0] << " config-file  image-file [image-file ...]\n";
    return 1;
  }

  // loop over file names
  for (int i = 2; i < argc; i++)
  {
    cv::Mat img = cv::imread( argv[i], CV_LOAD_IMAGE_COLOR );
    if ( ! img.data )
    {
      std::cerr << "Could not read image from file \"" << argv[i] << "\"\n";
      continue;
    }

    kwiver::SMQTK_Descriptor des;
    std::vector< double > results = des.ExtractSMQTK( img, argv[1] );

    std::cout << "Descriptor size: " << results.size()
              << std::endl;

    for ( unsigned i = 0; i < 50; i++)
    {
      std::cout << results.at(i) << " ";
    }
    std::cout << std::endl;

  } // end for

  return 0;
}
