// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_ARROWS_VISCL_FEATURE_SET_H_
#define KWIVER_ARROWS_VISCL_FEATURE_SET_H_

#include <vital/vital_config.h>
#include <arrows/viscl/kwiver_algo_viscl_export.h>

#include <vital/types/feature_set.h>

#include <viscl/core/buffer.h>
#include <viscl/core/image.h>

namespace kwiver {
namespace arrows {
namespace vcl {

/// A concrete feature set that wraps VisCL features
/**
  * A VisCL feature only has the location set
  * It is possible to get the smoothing scale but that value is not
  * saved on the GPU so would have to be provided externally
  */
class KWIVER_ALGO_VISCL_EXPORT feature_set
: public vital::feature_set
{
public:

  struct type
  {
    viscl::buffer features_;
    viscl::buffer numfeat_;
    viscl::image kptmap_;
  };

  /// Default Constructor
  feature_set() {}

  /// Constructor from VisCL data
  explicit feature_set(const type& viscl_features)
  : data_(viscl_features) {}

  /// Return the number of features in the set
  /**
    * Downloads the size from the GPU
    */
  virtual size_t size() const;

  /// Return a vector of feature shared pointers
  virtual std::vector<vital::feature_sptr> features() const;

  /// Return the underlying VisCL features data structure
  const type& viscl_features() const { return data_; }

protected:

  /// The VisCL feature point data
  type data_;
};

/// Convert any feature set to a VisCL data (upload if needed)
/**
  * viscl only cares about integer feature location, therefore will lose
  * info converting from vital feature set to viscl and back
  */
KWIVER_ALGO_VISCL_EXPORT feature_set::type
features_to_viscl(const vital::feature_set& features);

} // end namespace vcl
} // end namespace arrows
} // end namespace kwiver

#endif
