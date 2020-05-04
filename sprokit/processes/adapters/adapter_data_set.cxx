/*ckwg +29
 * Copyright 2016, 2019 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * \file
 * \brief Implementation for adapter_data_set class.
 */

#include "adapter_data_set.h"

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
  m_port_datum_set[port] = datum ;
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

} } // end namespace
