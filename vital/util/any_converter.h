/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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
 * \file \brief This file contains the any converter class
 */

#ifndef KWIVER_VITAL_NY_CONVERTER_H
#define KWIVER_VITAL_NY_CONVERTER_H

#include <vital/any.h>

#include <vector>
#include <memory>

namespace kwiver {
namespace vital {

// ==================================================================
/// Generic converter for any value to specific type.
/**
 * This class represents a list of converters that can convert from an
 * unspecified type in an kwiver::vital::any object to a specific
 * type.
 *
 * Example:
\code
any_converter<int> any_to_int;

any_to_int.add_converter<uint8_t>();  // add converter from uint8_t;
any_to_int.add_converter<float>();    // add converter from float;
\endcode
 *
 */
template <typename T>
class any_converter
{
public:

  // ------------------------------------------------------------------
  template < typename DEST >
  struct convert_base
  {
    convert_base() { }

    virtual ~convert_base() { }

    virtual bool can_convert( kwiver::vital::any const& data ) const = 0;
    virtual DEST convert( kwiver::vital::any const& data ) const = 0;
  };


  // ------------------------------------------------------------------
  template < typename DEST, typename SRC >
  struct converter
    : public convert_base< DEST >
  {
    converter() { }

    virtual ~converter() { }

    virtual bool can_convert( kwiver::vital::any const& data ) const
    {
      return data.type() == typeid( SRC );
    }

    virtual DEST convert( kwiver::vital::any const& data ) const
    {
      return static_cast< DEST > ( kwiver::vital::any_cast< SRC > ( data ) );
    }
  };


  typedef std::unique_ptr< convert_base< T > > converter_ptr;

  any_converter() {}
  virtual ~any_converter() {}

  /// Apply conversion to an kwiver::vital::any object.
  /**
   * This method looks for a compatible conversion method that will
   * allow the type in the parameter any to be converted to our
   * desired destination type.
   *
   * The list of registered converters is iterated over looking for a
   * converter that accepts the type in the any data. If one is found,
   * then that conversion is done. If no acceptable converter is
   * found, then an exception is thrown.
   *
   * @param data Value of inspecific type to be converted.
   *
   * @return Value from parameter converted to desired type.
   *
   * @throws bad_any_cast if the conversion is not successful.
   */
  T convert( kwiver::vital::any const& data ) const
  {
    const auto eix = m_converter_list.end();
    for (auto ix = m_converter_list.begin(); ix != eix; ix++)
    {
      if ( (*ix)->can_convert( data ) )
      {
        return (*ix)->convert( data );
      }
    } // end for

    // Throw exception
    throw kwiver::vital::bad_any_cast( data.type().name(), typeid(T).name() );
  }

  /// Test to see if conversion can be done.
  /**
   * This method checks to see if there is a suitable converter registered.
   *
   * @param data The any object to be converted.
   *
   * @return \b true if value can be converted, \b false otherwise.
   */
  bool can_convert( kwiver::vital::any const& data ) const
  {
    const auto eix = m_converter_list.end();
    for (auto ix = m_converter_list.begin(); ix != eix; ix++)
    {
      if ( (*ix)->can_convert( data ) )
      {
        return true;
      }
    } // end for

    return false;
  }

  /// Add converter based on from type.
  /**
   * Adds a new converter that handles a specific source type. The
   * types (DEST and SRC) must be convertable, either implicitly or
   * explicitly) for this to work
   *
   * tparam SRC Type from any to be converted.
   */
  template <typename SRC>
  void add_converter( )
  {
    m_converter_list.push_back( converter_ptr( new converter< T, SRC >() ) );
  }

private:
  std::vector< converter_ptr > m_converter_list;

}; // end class any_converter

} } // end namespace

#endif /* KWIVER_VITAL_NY_CONVERTER_H */
