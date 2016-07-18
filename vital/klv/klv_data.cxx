/*ckwg +29
 * Copyright 2015-2016 by Kitware, Inc.
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
 * \brief This file contains the implementation for the klv data class.
 */

#include <vital/klv/klv_data.h>

#include <algorithm>
#include <iomanip>


namespace kwiver {
namespace vital {

  klv_data
  ::klv_data()
    : m_key_offset( 0 ),
      key_len_( 0 ),
      m_value_offset( 0 ),
      m_value_len ( 0 )
  { }


klv_data
::klv_data(container_t const& raw_packet,
         std::size_t key_offset, std::size_t key_len,
         std::size_t m_value_offset, std::size_t value_len)
  : m_raw_data( raw_packet ),
    m_key_offset( key_offset ),
    key_len_( key_len ),
    m_value_offset( m_value_offset ),
    m_value_len ( value_len )
{ }


klv_data
::~klv_data()
{ }

std::size_t
klv_data
::key_size() const
{
  return this->key_len_;
}


std::size_t
klv_data
::value_size() const
{
  return this->m_value_len;
}


std::size_t
klv_data
::klv_size() const
{
  return this->m_raw_data.size();
}


klv_data::const_iterator_t
klv_data
::klv_begin() const
{
  return this->m_raw_data.begin();
}


klv_data::const_iterator_t
klv_data
::klv_end() const
{
  return this->m_raw_data.end();
}


klv_data::const_iterator_t
klv_data
::key_begin() const
{
  return this->m_raw_data.begin() + m_key_offset;
}


klv_data::const_iterator_t
klv_data
::key_end() const
{
  return this->m_raw_data.begin() + m_key_offset + key_len_;

}


klv_data::const_iterator_t
klv_data
::value_begin() const
{
  return this->m_raw_data.begin() + m_value_offset;
}


klv_data::const_iterator_t
klv_data
::value_end() const
{
  return this->m_raw_data.begin() + m_value_offset + m_value_len;
}


std::ostream & operator<<( std::ostream& str, klv_data const& obj )
{
  std::ostream::fmtflags f( str.flags() );
  int i = 0;
  // format the whole raw package
  str << "Raw packet: ";
  for (klv_data::const_iterator_t it = obj.klv_begin();
       it != obj.klv_end(); it++)
  {
    str << std::hex << std::setfill( '0' ) << std::setw( 2 ) << int(*it);
    if (i % 4 == 3) str << " "; i++;
  }

  str << "\n\nKey bytes: ";
  for (klv_data::const_iterator_t it = obj.key_begin();
       it != obj.key_end(); it++)
  {
    str << std::hex << std::setfill( '0' ) << std::setw( 2 ) << int(*it);
    if (i % 4 == 3) str << " "; i++;
  }

  str << "\n\nValue bytes: ";
  for (klv_data::const_iterator_t it = obj.value_begin();
       it != obj.value_end(); it++)
  {
    str << std::hex << std::setfill( '0' ) << std::setw( 2 ) << int(*it);
    if (i % 4 == 3) str << " "; i++;
  }

  str << std::endl;
  str.flags( f );
  return str;
}

} } // end namespace
