// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_ARROWS_VISCL_DESCRIPTOR_SET_H_
#define KWIVER_ARROWS_VISCL_DESCRIPTOR_SET_H_

#include <vital/vital_config.h>
#include <arrows/viscl/kwiver_algo_viscl_export.h>

#include <vital/types/descriptor_set.h>

#include <viscl/core/buffer.h>

namespace kwiver {
namespace arrows {
namespace vcl {

/// A concrete descriptor set that wraps VisCL descriptors.
class KWIVER_ALGO_VISCL_EXPORT descriptor_set
: public vital::descriptor_set
{
public:

  /// Default Constructor
  descriptor_set() {}

  /// Constructor from VisCL descriptors
  explicit descriptor_set(const viscl::buffer& viscl_descriptors)
  : data_(viscl_descriptors) {}

  /// Return the number of descriptor in the set
  virtual size_t size() const { return data_.len(); }

  /// Return a vector of descriptor shared pointers
  /**
    * Warning: These descriptors must be matched by hamming distance
    */
  virtual std::vector<vital::descriptor_sptr> descriptors() const;

  /// Return the native VisCL descriptors structure
  const viscl::buffer& viscl_descriptors() const { return data_; }

protected:

  /// The handle to a VisCL set of descriptors
  viscl::buffer data_;
};

/// Convert a descriptor set to a VisCL descriptor set must be <int,4>
KWIVER_ALGO_VISCL_EXPORT viscl::buffer
descriptors_to_viscl(const vital::descriptor_set& desc_set);

} // end namespace vcl
} // end namespace arrows
} // end namespace kwiver

#endif
