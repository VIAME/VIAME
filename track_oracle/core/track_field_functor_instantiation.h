/*ckwg +5
 * Copyright 2014-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef TRACK_FIELD_FUNCTOR_INSTANCES_H
#define TRACK_FIELD_FUNCTOR_INSTANCES_H

#include <vital/vital_config.h>
#include <track_oracle/core/track_oracle_export.h>

#define TRACK_FIELD_FUNCTOR_INSTANCES(T) \
  template TRACK_ORACLE_EXPORT class ::kwiver::track_oracle::track_field_functor<T>;


#endif
