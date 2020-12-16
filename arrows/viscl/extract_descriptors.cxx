// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "extract_descriptors.h"

#include <arrows/viscl/image_container.h>
#include <arrows/viscl/feature_set.h>
#include <arrows/viscl/descriptor_set.h>

#include <viscl/tasks/BRIEF.h>

namespace kwiver {
namespace arrows {
namespace vcl {

/// Private implementation class
class extract_descriptors::priv
{
public:
  /// Constructor
  priv()
  {
  }

  viscl::brief<10> brief;
};

/// Constructor
extract_descriptors
::extract_descriptors()
: d_(new priv)
{
}

/// Destructor
extract_descriptors
::~extract_descriptors()
{
}

/// Extract from the image a descriptor corresoponding to each feature
vital::descriptor_set_sptr
extract_descriptors
::extract(vital::image_container_sptr image_data,
          vital::feature_set_sptr features,
          vital::image_container_sptr /* image_mask */) const
{
  if( !image_data || !features )
  {
    return vital::descriptor_set_sptr();
  }

  viscl::image img = vcl::image_container_to_viscl(*image_data);
  vcl::feature_set::type fs = vcl::features_to_viscl(*features);
  viscl::buffer descriptors;
  d_->brief.compute_descriptors(img, fs.features_, features->size(), descriptors);
  return vital::descriptor_set_sptr(new vcl::descriptor_set(descriptors));
}

} // end namespace vcl
} // end namespace arrows
} // end namespace kwiver
