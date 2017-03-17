/*ckwg +5
 * Copyright 2014-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef INCL_TRACK_FIELD_INSTANTIATION_H
#define INCL_TRACK_FIELD_INSTANTIATION_H

#include <track_oracle/core/track_field.txx>
#include <track_oracle/core/track_field_io_proxy.txx>

#define TRACK_FIELD_INSTANCES_GENERAL(T) \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T>::track_field( const kwiver::track_oracle::track_field<T>& );     \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T>::~track_field(); \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T>& kwiver::track_oracle::track_field<T>::operator=( const kwiver::track_oracle::track_field<T>& ); \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T>::Type kwiver::track_oracle::track_field<T>::operator()() const; \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T>::Type& kwiver::track_oracle::track_field<T>::operator()(); \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T>::Type kwiver::track_oracle::track_field<T>::operator()( const kwiver::track_oracle::oracle_entry_handle_type& ) const; \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T>::Type& kwiver::track_oracle::track_field<T>::operator()( const kwiver::track_oracle::oracle_entry_handle_type&  ); \
  template TRACK_FIELD_EXPORT void kwiver::track_oracle::track_field<T>::remove_at_row( const kwiver::track_oracle::oracle_entry_handle_type& ); \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::oracle_entry_handle_type kwiver::track_oracle::track_field<T>::lookup( const kwiver::track_oracle::track_field<T>::Type&, kwiver::track_oracle::domain_handle_type ) ; \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::oracle_entry_handle_type kwiver::track_oracle::track_field<T>::lookup( const kwiver::track_oracle::track_field<T>::Type&, const kwiver::track_oracle::track_handle_type& ) ; \
  template TRACK_FIELD_EXPORT bool kwiver::track_oracle::track_field<T>::exists( const oracle_entry_handle_type& ) const; \
  template TRACK_FIELD_EXPORT bool kwiver::track_oracle::track_field<T>::exists( void ) const; \
  template TRACK_FIELD_EXPORT std::pair< bool, kwiver::track_oracle::track_field<T>::Type > kwiver::track_oracle::track_field<T>::get( kwiver::track_oracle::oracle_entry_handle_type ) const; \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T>* kwiver::track_oracle::track_field<T>::clone() const; \
  template TRACK_FIELD_EXPORT void kwiver::track_oracle::track_field<T>::copy_value( const kwiver::track_oracle::oracle_entry_handle_type&, const kwiver::track_oracle::oracle_entry_handle_type& ) const; \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::field_handle_type kwiver::track_oracle::track_field<T>::lookup_or_create_element_store( const std::string & ); \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field_io_proxy< kwiver::track_oracle::track_field<T>::Type > kwiver::track_oracle::track_field<T>::io() const; \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field_io_proxy< kwiver::track_oracle::track_field<T>::Type > kwiver::track_oracle::track_field<T>::io( const kwiver::track_oracle::oracle_entry_handle_type& ) const; \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field_io_proxy< kwiver::track_oracle::track_field<T>::Type > kwiver::track_oracle::track_field<T>::io_fmt( const kwiver::track_oracle::track_field<T>::Type& ) const;
// seems to cause errors building track_oracle project win64
// template std::ostream& kwiver::track_oracle::operator<< <kwiver::track_oracle::track_field<T>::Type> ( std::ostream&, const kwiver::track_oracle::track_field_io_proxy<kwiver::track_oracle::track_field<T>::Type>& );

#define TRACK_FIELD_INSTANCES_GENERAL_DEBUG(T) \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T>::track_field( const kwiver::track_oracle::track_field<T>& );     \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T>::~track_field(); \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T>& kwiver::track_oracle::track_field<T>::operator=( const kwiver::track_oracle::track_field<T>& ); \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T>::Type kwiver::track_oracle::track_field<T>::operator()() const; \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T>::Type& kwiver::track_oracle::track_field<T>::operator()(); \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T>::Type kwiver::track_oracle::track_field<T>::operator()( const kwiver::track_oracle::oracle_entry_handle_type& ) const; \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T>::Type& kwiver::track_oracle::track_field<T>::operator()( const kwiver::track_oracle::oracle_entry_handle_type&  ); \
  template TRACK_FIELD_EXPORT void kwiver::track_oracle::track_field<T>::remove_at_row( const kwiver::track_oracle::oracle_entry_handle_type& ); \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::oracle_entry_handle_type kwiver::track_oracle::track_field<T>::lookup( const kwiver::track_oracle::track_field<T>::Type&, kwiver::track_oracle::domain_handle_type ) ; \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::oracle_entry_handle_type kwiver::track_oracle::track_field<T>::lookup( const kwiver::track_oracle::track_field<T>::Type&, const kwiver::track_oracle::track_handle_type& ) ; \
  template TRACK_FIELD_EXPORT bool kwiver::track_oracle::track_field<T>::exists( const oracle_entry_handle_type& ) const; \
  template TRACK_FIELD_EXPORT bool kwiver::track_oracle::track_field<T>::exists( void ) const; \
  template TRACK_FIELD_EXPORT std::pair< bool, kwiver::track_oracle::track_field<T>::Type > kwiver::track_oracle::track_field<T>::get( kwiver::track_oracle::oracle_entry_handle_type ) const; \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T>* kwiver::track_oracle::track_field<T>::clone() const; \
  template TRACK_FIELD_EXPORT void kwiver::track_oracle::track_field<T>::copy_value( const kwiver::track_oracle::oracle_entry_handle_type&, const kwiver::track_oracle::oracle_entry_handle_type& ) const; \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::field_handle_type kwiver::track_oracle::track_field<T>::lookup_or_create_element_store( const std::string & ); \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field_io_proxy< kwiver::track_oracle::track_field<T>::Type > kwiver::track_oracle::track_field<T>::io() const; \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field_io_proxy< kwiver::track_oracle::track_field<T>::Type > kwiver::track_oracle::track_field<T>::io( const kwiver::track_oracle::oracle_entry_handle_type& ) const; \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field_io_proxy< kwiver::track_oracle::track_field<T>::Type > kwiver::track_oracle::track_field<T>::io_fmt( const kwiver::track_oracle::track_field<T>::Type& ) const;
// seems to cause problems building track_oracle project on win64
// template std::ostream& kwiver::track_oracle::operator<< <kwiver::track_oracle::track_field<T>::Type> ( std::ostream&, const kwiver::track_oracle::track_field_io_proxy<kwiver::track_oracle::track_field<T>::Type>& );


#define TRACK_FIELD_INSTANCES_DATA_TERM_SPECIAL_OUTPUT(T) \
  TRACK_FIELD_INSTANCES_GENERAL(T) \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T>::track_field();                        \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T>::track_field( kwiver::track_oracle::track_field_host * );

#define TRACK_FIELD_INSTANCES_OLD_STYLE_SPECIAL_OUTPUT(T) \
  TRACK_FIELD_INSTANCES_GENERAL(T) \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T>::track_field( const std::string& ); \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T>::track_field( const std::string&, kwiver::track_oracle::track_field_host* );

#define TRACK_FIELD_INSTANCES_DATA_TERM_DEFAULT_OUTPUT(T) \
  TRACK_FIELD_INSTANCES_GENERAL(T) \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T>::track_field();                        \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T>::track_field( kwiver::track_oracle::track_field_host * ); \
  template TRACK_FIELD_EXPORT std::ostream& kwiver::track_oracle::operator<< < T >( std::ostream&, const kwiver::track_oracle::track_field< T >& );

#define TRACK_FIELD_INSTANCES_OLD_STYLE_DEFAULT_OUTPUT(T) \
  TRACK_FIELD_INSTANCES_GENERAL(T) \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T>::track_field( const std::string& ); \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T>::track_field( const std::string&, kwiver::track_oracle::track_field_host* ); \
  template TRACK_FIELD_EXPORT std::ostream& kwiver::track_oracle::operator<< < T >( std::ostream&, const kwiver::track_oracle::track_field< T >& );

#define TRACK_FIELD_INSTANCES_OLD_STYLE_DEFAULT_OUTPUT_DEBUG(T) \
  TRACK_FIELD_INSTANCES_GENERAL_DEBUG(T) \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T>::track_field( const std::string& ); \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T>::track_field( const std::string&, kwiver::track_oracle::track_field_host* ); \
  template TRACK_FIELD_EXPORT std::ostream& kwiver::track_oracle::operator<< < T >( std::ostream&, const kwiver::track_oracle::track_field< T >& );

// argh, commas

#define TF_MACRO_COMMA ,

#define TRACK_FIELD_INSTANCES_OLD_STYLE_SPECIAL_OUTPUT_COMMA(T, T2)        \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T, T2>::track_field( const std::string& );    \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T, T2>::track_field( const std::string&, kwiver::track_oracle::track_field_host* ); \
  template TRACK_FIELD_EXPORT std::ostream& kwiver::track_oracle::operator<< < T, T2 >( std::ostream&, const kwiver::track_oracle::track_field< T, T2 >& );

#undef TF_MACRO_COMMA

#define TRACK_FIELD_INSTANCES_DATA_TERM_IO(T) \
 template std::ostream& kwiver::track_oracle::operator<< <kwiver::track_oracle::track_field<T>::Type> ( std::ostream&, const kwiver::track_oracle::track_field_io_proxy<kwiver::track_oracle::track_field<T>::Type>& );


#define TRACK_FIELD_INSTANCES_DATA_TERM(T) \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T>::track_field(); \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T>::~track_field(); \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T>::track_field( kwiver::track_oracle::track_field_host * ); \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T>::track_field( const kwiver::track_oracle::track_field<T>& );     \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T>& kwiver::track_oracle::track_field<T>::operator=( const kwiver::track_oracle::track_field<T>& ); \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T>::Type kwiver::track_oracle::track_field<T>::operator()() const; \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T>::Type& kwiver::track_oracle::track_field<T>::operator()(); \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T>::Type kwiver::track_oracle::track_field<T>::operator()( const kwiver::track_oracle::oracle_entry_handle_type& ) const; \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T>::Type& kwiver::track_oracle::track_field<T>::operator()( const kwiver::track_oracle::oracle_entry_handle_type&  ); \
  template TRACK_FIELD_EXPORT void kwiver::track_oracle::track_field<T>::remove_at_row( const kwiver::track_oracle::oracle_entry_handle_type& ); \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::oracle_entry_handle_type kwiver::track_oracle::track_field<T>::lookup( const kwiver::track_oracle::track_field<T>::Type&, kwiver::track_oracle::domain_handle_type ) ; \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::oracle_entry_handle_type kwiver::track_oracle::track_field<T>::lookup( const kwiver::track_oracle::track_field<T>::Type&, const kwiver::track_oracle::track_handle_type& ) ; \
  template TRACK_FIELD_EXPORT bool kwiver::track_oracle::track_field<T>::exists( const oracle_entry_handle_type& ) const; \
  template TRACK_FIELD_EXPORT bool kwiver::track_oracle::track_field<T>::exists( void ) const; \
  template TRACK_FIELD_EXPORT std::pair< bool, kwiver::track_oracle::track_field<T>::Type > kwiver::track_oracle::track_field<T>::get( kwiver::track_oracle::oracle_entry_handle_type ) const; \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field<T>* kwiver::track_oracle::track_field<T>::clone() const; \
  template TRACK_FIELD_EXPORT void kwiver::track_oracle::track_field<T>::copy_value( const kwiver::track_oracle::oracle_entry_handle_type&, const kwiver::track_oracle::oracle_entry_handle_type& ) const; \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field_io_proxy< kwiver::track_oracle::track_field<T>::Type > kwiver::track_oracle::track_field<T>::io() const; \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field_io_proxy< kwiver::track_oracle::track_field<T>::Type > kwiver::track_oracle::track_field<T>::io( const kwiver::track_oracle::oracle_entry_handle_type& ) const; \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::track_field_io_proxy< kwiver::track_oracle::track_field<T>::Type > kwiver::track_oracle::track_field<T>::io_fmt( const kwiver::track_oracle::track_field<T>::Type& ) const; \
  template TRACK_FIELD_EXPORT kwiver::track_oracle::field_handle_type kwiver::track_oracle::track_field<T>::lookup_or_create_element_store( const std::string & ); \
  template TRACK_FIELD_EXPORT std::ostream& kwiver::track_oracle::operator<< ( std::ostream&, const kwiver::track_oracle::track_field<T>&  );

#endif
