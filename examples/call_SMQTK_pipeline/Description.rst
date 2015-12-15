
Calling SMQTK descriptors
=========================

The C++ interface to SMQTK descriptors is through the SMQTK_Descriptor class (SMQTK_Descriptor.h).
The class provides a single method to determine the descriptor, which is described as follows:

``std::vector< double > ExtractSMQTK(  cv::Mat cv_img, std::string const& config_file );``

The parameters are an image in OpenCV format and the name of the SMQTK descriptor configuration file.
The contents of the configuration file will depend on which descriptor is to be used. An example of
a configuration for the caffenet configuration follows::

 {
    "blvc_reference_caffenet_model": "/home/etri/projects/smqtk/source/data/caffenet/bvlc_reference_caffenet.caffemodel",
    "image_mean_binary": "/home/etri/projects/smqtk/source/data/caffenet/imagenet_mean.binaryproto",
    "gpu_batch_size": 100
 }

The ExtractSMQTK() method is synchronous in that it will return with the descriptor vector even though the
descriptor calculation may be multi-threaded.


Usage
-----

Sample code for using the SMQTK_Descriptor class is included in the SMQTK_Descriptor_test.cxx file.

The following code is needed to use this class::

  #include "SMQTK_Descriptor.h"

  kwiver::SMQTK_Descriptor des; // Allocate object
  std::vector< double > results = des.ExtractSMQTK( img, file_name ); // process image
