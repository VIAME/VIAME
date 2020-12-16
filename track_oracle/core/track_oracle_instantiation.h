// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_TRACK_ORACLE_INSTANCES_H
#define INCL_TRACK_ORACLE_INSTANCES_H

#include <track_oracle/core/track_oracle_core_impl.txx>
#include <track_oracle/core/track_oracle_core.txx>

#define TRACK_ORACLE_INSTANCES(T) \
  template TRACK_ORACLE_CORE_EXPORT kwiver::track_oracle::field_handle_type kwiver::track_oracle::track_oracle_core_impl::unlocked_create_element<T>( const kwiver::track_oracle::element_descriptor& e ); \
  template TRACK_ORACLE_CORE_EXPORT kwiver::track_oracle::field_handle_type kwiver::track_oracle::track_oracle_core::create_element<T>( const kwiver::track_oracle::element_descriptor& e ); \
  template TRACK_ORACLE_CORE_EXPORT T& kwiver::track_oracle::track_oracle_core_impl::unlocked_get_field<T>( kwiver::track_oracle::oracle_entry_handle_type track, kwiver::track_oracle::field_handle_type field ); \
  template TRACK_ORACLE_CORE_EXPORT T& kwiver::track_oracle::track_oracle_core::get_field<T>( kwiver::track_oracle::oracle_entry_handle_type track, kwiver::track_oracle::field_handle_type field ); \
  template TRACK_ORACLE_CORE_EXPORT std::pair< bool, T > kwiver::track_oracle::track_oracle_core::get<T>( const kwiver::track_oracle::oracle_entry_handle_type& track, const kwiver::track_oracle::field_handle_type& field ); \
  template TRACK_ORACLE_CORE_EXPORT std::pair< bool, T > kwiver::track_oracle::track_oracle_core_impl::get<T>( kwiver::track_oracle::oracle_entry_handle_type track, kwiver::track_oracle::field_handle_type field ); \
  template TRACK_ORACLE_CORE_EXPORT kwiver::track_oracle::oracle_entry_handle_type kwiver::track_oracle::track_oracle_core::lookup<T>( kwiver::track_oracle::field_handle_type field, const T& val, kwiver::track_oracle::domain_handle_type domain ); \
  template TRACK_ORACLE_CORE_EXPORT void kwiver::track_oracle::track_oracle_core::remove_field<T>( kwiver::track_oracle::oracle_entry_handle_type row, kwiver::track_oracle::field_handle_type field );\
  template TRACK_ORACLE_CORE_EXPORT std::pair< std::map<kwiver::track_oracle::oracle_entry_handle_type, T>*, T> kwiver::track_oracle::track_oracle_core_impl::lookup_table<T>( kwiver::track_oracle::field_handle_type field ); \
  template TRACK_ORACLE_CORE_EXPORT T& kwiver::track_oracle::track_oracle_core_impl::get_field<T>( kwiver::track_oracle::oracle_entry_handle_type track, kwiver::track_oracle::field_handle_type field ); \
  template TRACK_ORACLE_CORE_EXPORT void kwiver::track_oracle::track_oracle_core_impl::remove_field<T>( kwiver::track_oracle::oracle_entry_handle_type row, kwiver::track_oracle::field_handle_type field ); \
  template TRACK_ORACLE_CORE_EXPORT kwiver::track_oracle::oracle_entry_handle_type kwiver::track_oracle::track_oracle_core_impl::lookup<T>( kwiver::track_oracle::field_handle_type field, const T& val, kwiver::track_oracle::domain_handle_type domain );

#endif
