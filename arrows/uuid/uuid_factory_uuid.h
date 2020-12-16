// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface to uid factory
 */

#ifndef KWIVER_ARROWS_UUID_FACTORY_H
#define KWIVER_ARROWS_UUID_FACTORY_H

#include <arrows/uuid/kwiver_algo_uuid_export.h>
#include <vital/algo/uuid_factory.h>

namespace kwiver {
namespace arrows {
namespace uuid {

class KWIVER_ALGO_UUID_EXPORT uuid_factory_uuid
  : public vital::algo::uuid_factory
{
public:
  uuid_factory_uuid();
  virtual ~uuid_factory_uuid();

  virtual void set_configuration(vital::config_block_sptr config);
  virtual bool check_configuration(vital::config_block_sptr config) const;

  // Main method to generate UUID's
  virtual kwiver::vital::uid create_uuid();

private:
  class priv;
  std::unique_ptr< priv > d;
};

} } } // end namespace

#endif /* KWIVER_ARROWS_UUID_FACTORY_H */
