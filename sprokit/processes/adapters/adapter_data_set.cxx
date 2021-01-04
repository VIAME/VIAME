// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation for adapter_data_set class.
 */

#include "adapter_data_set.h"

#include <kwiver_type_traits.h>

namespace kwiver {
namespace adapter {

namespace {

// Hack to allow std::make_shared<> work when CTOR is private.
struct local_ads : public adapter_data_set {
  local_ads( data_set_type type ) : adapter_data_set(type) {}
};

} // end namespace

// ------------------------------------------------------------------
adapter_data_set
::adapter_data_set( data_set_type type )
  :m_set_type( type )
{ }

adapter_data_set
::~adapter_data_set()
{ }

// ------------------------------------------------------------------
adapter_data_set_t
adapter_data_set
::create( data_set_type type )
{
  adapter_data_set_t set = std::make_shared<local_ads>( type );
  return set;
}

// ------------------------------------------------------------------
void
adapter_data_set
::add_datum( sprokit::process::port_t const& port, sprokit::datum_t const& datum )
{
  m_port_datum_set[port] = datum;
}

// ------------------------------------------------------------------
bool
adapter_data_set
::empty() const
{
  return m_port_datum_set.empty();
}

// ------------------------------------------------------------------
kwiver::adapter::adapter_data_set::datum_map_t::iterator
adapter_data_set
::begin()
{
  return m_port_datum_set.begin();
}

// ------------------------------------------------------------------
kwiver::adapter::adapter_data_set::datum_map_t::const_iterator
adapter_data_set
::begin() const
{
  return m_port_datum_set.begin();
}

// ------------------------------------------------------------------
kwiver::adapter::adapter_data_set::datum_map_t::const_iterator
adapter_data_set
::cbegin() const
{
  return m_port_datum_set.begin();
}

// ------------------------------------------------------------------
kwiver::adapter::adapter_data_set::datum_map_t::iterator
adapter_data_set
::end()
{
  return m_port_datum_set.end();
}

// ------------------------------------------------------------------
kwiver::adapter::adapter_data_set::datum_map_t::const_iterator
adapter_data_set
::end() const
{
  return m_port_datum_set.end();
}

// ------------------------------------------------------------------
kwiver::adapter::adapter_data_set::datum_map_t::const_iterator
adapter_data_set
::cend() const
{
  return m_port_datum_set.end();
}

// ------------------------------------------------------------------
kwiver::adapter::adapter_data_set::datum_map_t::const_iterator
adapter_data_set
::find( sprokit::process::port_t const& port ) const
{
  return m_port_datum_set.find( port );
}

// ------------------------------------------------------------------
bool
kwiver::adapter::adapter_data_set::is_end_of_data() const
{
  return (m_set_type == end_of_input);
}

// ------------------------------------------------------------------
adapter_data_set::data_set_type
kwiver::adapter::adapter_data_set::type() const
{
  return m_set_type;
}

// ------------------------------------------------------------------
size_t
adapter_data_set::size() const
{
  return m_port_datum_set.size();
}

// ------------------------------------------------------------------
template <typename T>
void adapter_data_set::add_value(::sprokit::process::port_t const& port, T const& val)
{
  m_port_datum_set[port] = ::sprokit::datum::new_datum<T>(val);
}

// ------------------------------------------------------------------
template<typename T>
T adapter_data_set::get_port_data( ::sprokit::process::port_t const& port )
{
  auto it = this->find( port );
  if ( it == this->end() )
  {
    throw std::runtime_error( "Data for port \"" + port + "\" is not in the adapter_data_set." );
  }
  return it->second->get_datum<T>();
}

// ------------------------------------------------------------------
#define INSTANTIATE_ADS_ADD_VALUE(T) \
  template KWIVER_ADAPTER_EXPORT \
  void \
  adapter_data_set::add_value(::sprokit::process::port_t const& port, T const& val);

#define INSTANTIATE_ADS_GET_PORT_DATA(T) \
  template KWIVER_ADAPTER_EXPORT \
  T \
  adapter_data_set::get_port_data(::sprokit::process::port_t const& port);

#define INSTANTIATE_ADS_ADD_GET_VALUE(T) \
  INSTANTIATE_ADS_ADD_VALUE(T) \
  INSTANTIATE_ADS_GET_PORT_DATA(T)

INSTANTIATE_ADS_ADD_GET_VALUE(int);
INSTANTIATE_ADS_ADD_GET_VALUE(float);
INSTANTIATE_ADS_ADD_GET_VALUE(double);
INSTANTIATE_ADS_ADD_GET_VALUE(bool);

INSTANTIATE_ADS_ADD_GET_VALUE(kwiver::vital::any);
INSTANTIATE_ADS_ADD_GET_VALUE(kwiver::vital::bounding_box_d);
INSTANTIATE_ADS_ADD_GET_VALUE(kwiver::vital::timestamp);
INSTANTIATE_ADS_ADD_GET_VALUE(kwiver::vital::geo_polygon);
INSTANTIATE_ADS_ADD_GET_VALUE(kwiver::vital::image_container_sptr);
INSTANTIATE_ADS_ADD_GET_VALUE(kwiver::vital::image_container_set_sptr);
INSTANTIATE_ADS_ADD_GET_VALUE(kwiver::vital::feature_set_sptr);
INSTANTIATE_ADS_ADD_GET_VALUE(kwiver::vital::database_query_sptr);
INSTANTIATE_ADS_ADD_GET_VALUE(kwiver::vital::descriptor_set_sptr);
INSTANTIATE_ADS_ADD_GET_VALUE(kwiver::vital::descriptor_request_sptr);
INSTANTIATE_ADS_ADD_GET_VALUE(kwiver::vital::query_result_set_sptr);
INSTANTIATE_ADS_ADD_GET_VALUE(kwiver::vital::iqr_feedback_sptr);
INSTANTIATE_ADS_ADD_GET_VALUE(kwiver::vital::string_t);
INSTANTIATE_ADS_ADD_GET_VALUE(kwiver::vital::string_vector_sptr);
INSTANTIATE_ADS_ADD_GET_VALUE(kwiver::vital::track_set_sptr);
INSTANTIATE_ADS_ADD_GET_VALUE(kwiver::vital::feature_track_set_sptr);
INSTANTIATE_ADS_ADD_GET_VALUE(kwiver::vital::object_track_set_sptr);
INSTANTIATE_ADS_ADD_GET_VALUE(kwiver::vital::double_vector_sptr);
INSTANTIATE_ADS_ADD_GET_VALUE(kwiver::vital::detected_object_set_sptr);
INSTANTIATE_ADS_ADD_GET_VALUE(kwiver::vital::track_descriptor_set_sptr);
INSTANTIATE_ADS_ADD_GET_VALUE(kwiver::vital::matrix_d);
INSTANTIATE_ADS_ADD_GET_VALUE(kwiver::vital::f2f_homography);
INSTANTIATE_ADS_ADD_GET_VALUE(kwiver::vital::metadata_vector);
INSTANTIATE_ADS_ADD_GET_VALUE(kwiver::vital::uid);

INSTANTIATE_ADS_ADD_GET_VALUE(std::shared_ptr<std::vector<unsigned char>>);
INSTANTIATE_ADS_ADD_GET_VALUE(kwiver::vital::string_sptr);

#undef INSTANTIATE_ADS_ADD_VALUE
#undef INSTANTIATE_ADS_GET_PORT_DATA
#undef INSTANTIATE_ADS_ADD_GET_VALUE

} } // end namespace
