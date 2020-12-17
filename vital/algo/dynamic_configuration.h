// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef DYNAMIC_CONFIGURATION_H
#define DYNAMIC_CONFIGURATION_H

#include <vital/algo/algorithm.h>

namespace kwiver {
namespace vital {
namespace algo {

/// Abstract algorithm for getting dynamic configuration values from
/// an external source.
/**
 * This class represents an interface to an external source of
 * configuration values. A typical application would be an external
 * U.I. control that is desired to control the performance of an
 * algorithm by varying some of its configuration values.
 */
class VITAL_ALGO_EXPORT dynamic_configuration :
    public kwiver::vital::algorithm_def< dynamic_configuration >
{
public:
  static std::string static_type_name() { return "dynamic_configuration"; }

  virtual void set_configuration( config_block_sptr config ) = 0;
  virtual bool check_configuration( config_block_sptr config ) const = 0;

  /// Return dynamic configuration values
  /**
   * This method returns dynamic configuration values. a valid config
   * block is returned even if there are not values being returned.
   */
  virtual config_block_sptr get_dynamic_configuration() = 0;

protected:
  dynamic_configuration();
};

/// Shared pointer for generic dynamic_configuration definition type.
typedef std::shared_ptr< dynamic_configuration > dynamic_configuration_sptr;

}
}
}     // end namespace

#endif // DYNAMIC_CONFIGURATION_H
