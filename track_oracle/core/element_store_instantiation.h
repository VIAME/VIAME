// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_ELEMENT_STORE_INSTANCES_H
#define INCL_ELEMENT_STORE_INSTANCES_H

#include <track_oracle/core/element_store.txx>

#define ELEMENT_STORE_INSTANCES(T) \
  template ELEMENT_STORE_EXPORT void ::kwiver::track_oracle::element_store<T>::set_io_handler( ::kwiver::track_oracle::kwiver_io_base<T>* ); \
  template ELEMENT_STORE_EXPORT ::kwiver::track_oracle::kwiver_io_base<T>* ::kwiver::track_oracle::element_store<T>::get_io_handler() const; \
  template ELEMENT_STORE_EXPORT void ::kwiver::track_oracle::element_store<T>::set_default_value(const T&);

#endif
