/*ckwg +5
 * Copyright 2014-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef INCL_ELEMENT_STORE_INSTANCES_H
#define INCL_ELEMENT_STORE_INSTANCES_H

#include <vital/vital_config.h>
#include <track_oracle/core/track_oracle_export.h>

#include <track_oracle/core/element_store.txx>

#define ELEMENT_STORE_INSTANCES(T) \
  template TRACK_ORACLE_EXPORT void ::kwiver::track_oracle::element_store<T>::set_io_handler( ::kwiver::track_oracle::kwiver_io_base<T>* ); \
  template TRACK_ORACLE_EXPORT ::kwiver::track_oracle::kwiver_io_base<T>* ::kwiver::track_oracle::element_store<T>::get_io_handler() const;


#endif
