// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_ARROWS_VISCL_EXTRACT_DESCRIPTORS_H_
#define KWIVER_ARROWS_VISCL_EXTRACT_DESCRIPTORS_H_

#include <arrows/viscl/kwiver_algo_viscl_export.h>

#include <vital/algo/extract_descriptors.h>

namespace kwiver {
namespace arrows {
namespace vcl {

/// An class for extracting feature descriptors using VisCL
class KWIVER_ALGO_VISCL_EXPORT extract_descriptors
: public vital::algo::extract_descriptors
{
public:
  /// Default Constructor
  extract_descriptors();

  /// Destructor
  virtual ~extract_descriptors();

  // No configuration for this class yet TODO: eventually descriptor size
  virtual void set_configuration(vital::config_block_sptr /*config*/) { }
  virtual bool check_configuration(vital::config_block_sptr /*config*/) const { return true; }

  /// Extract from the image a descriptor corresoponding to each feature
  /** \param image_data contains the image data to process
    * \param features the feature locations at which descriptors are extracted
    * \returns a set of feature descriptors
    */
  virtual vital::descriptor_set_sptr
  extract(vital::image_container_sptr image_data,
          vital::feature_set_sptr features,
          vital::image_container_sptr image_mask = vital::image_container_sptr()) const;

private:
  /// private implementation class
  class priv;
  const std::unique_ptr<priv> d_;
};

} // end namespace vcl
} // end namespace arrows
} // end namespace kwiver

#endif
