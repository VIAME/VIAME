// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface to uid factory
 */

#ifndef VITAL_ALGO_UUID_FACTORY_H
#define VITAL_ALGO_UUID_FACTORY_H

#include <vital/vital_config.h>
#include <vital/algo/algorithm.h>
#include <vital/types/uid.h>

namespace kwiver {
namespace vital {
namespace algo {

/// Abstract base class for creating uuid's
/**
 *
 */
class VITAL_ALGO_EXPORT uuid_factory
  : public kwiver::vital::algorithm_def< uuid_factory >
{
public:
  /// Return the name of this algorithm
  static std::string static_type_name() { return "uuid_factory"; }

  virtual uid create_uuid() = 0;

protected:
  uuid_factory();
};

typedef std::shared_ptr< uuid_factory > uuid_factory_sptr;

} } } // end namespace

#endif /* VITAL_ALGO_UUID_FACTORY_H */
