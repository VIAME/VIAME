/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "process_cluster.h"

#include "process_exception.h"

/**
 * \file process_cluster.cxx
 *
 * \brief Implementation for \link vistk::process_cluster process cluster\endlink.
 */

namespace vistk
{

process::property_t const process_cluster::property_cluster = process::property_t("_cluster");

class process_cluster::priv
{
  public:
    priv();
    ~priv();
};

process_cluster
::process_cluster(config_t const& config)
  : process(config)
{
}

process_cluster
::~process_cluster()
{
}

void
process_cluster
::_configure()
{
}

void
process_cluster
::_init()
{
}

void
process_cluster
::_reset()
{
}

void
process_cluster
::_step()
{
  throw process_exception();
}

process::properties_t
process_cluster
::_properties() const
{
  properties_t base_properties = process::_properties();

  base_properties.insert(property_cluster);

  return base_properties;
}

process_cluster::priv
::priv()
{
}

process_cluster::priv
::~priv()
{
}

}
