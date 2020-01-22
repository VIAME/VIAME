/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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

#ifndef VITAL_UTIL_ENUM_CONVERTER_H
#define VITAL_UTIL_ENUM_CONVERTER_H


#include <vector>
#include <utility>
#include <string>
#include <iostream>
#include <sstream>
#include <exception>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------
/**
 * @brief Convert enum to string or string to value.
 *
 * This class converts between a string representation of an enum
 * element and its integer value. This class is intended to provide
 * uniform handling of enums for configuring algorithms and processes.
 *
 * The following example shows how to use the base class as a
 * converter. In this case, the class must be instantiated and
 * initialized with the conversion entries.
 *
\code
enum numbers { one = 1, two, three, four, five };

enum_converter<numbers> ec({
        // name,   value
        { "one",   one },
        { "two",   two },
        { "three", three },
        { "four",  four },
        { "five",  five }
    } );

std::string name = ec.to_string( one );
\endcode
 *
 * The following example shows how to use the ENUM_CONVERTER() macro to
 * create a enum conversion type that can be used without expressly
 * instantiating an instance.
 *
\code
// Converter implemented as a derived type using MACRO helper
ENUM_CONVERTER( my_ec, numbers,
      { "ONE", one },
      { "TWO", two },
      { "THREE", three },
      { "FOUR", four },
      { "FIVE", five }
)

// Using converter directly from type name
std::cout << my_ec().to_string(three)  << std::endl;
\endcode
 *
 * The following example code fragments show how a converter can be
 * used in an algorithm to establish a configuration entry. In this
 * case, the default value from \c d->method is converted into a
 * string as the default value. In addition, the list of available
 * names is added to the end of the documentation string. Also shows
 * is how to use the get_enum_value() templated method to convert a
 * string from a config entry into the associated enum value.
 *
\code
// Define the enum converter
ENUM_CONVERTER( method_converter, method_t,
                { "EST_7_POINT",   EST_7_POINT },
                { "EST_8_POINT",   EST_8_POINT }
)

  config->set_value("method", method_converter().to_string( d->method ),
                    "Fundamental matrix estimation method to use. "
                    "(Note: does not include RANSAC).  Choices are: "
                    + method_converter().element_name_string() );

  // Convert config entry from string to enum value
  d->method = config->get_enum_value< method_converter >( "method" );
\endcode
 *
 * The following example code fragment shows how a configuration entry
 * in a process can be converted from a config string to an internal code.
 *
\code
  enum error_mode_t {           // Error handling modes
    ERROR_ABORT = 1,
    ERROR_SKIP
  };

  // Define the enum converter
  ENUM_CONVERTER( mode_converter, error_mode_t,
                  { "abort",   ERROR_ABORT },
                  { "skip",    ERROR_SKIP }
    )

  // Get value from process config and convert into enum value
  std::string mode = config_value_using_trait( error_mode );
  d->m_config_error_mode = priv::mode_converter().from_string( mode );
\endcode
 *
 *
 * @tparam T Type of the enum being converted
 */
template < typename T >
struct enum_converter
{
  typedef T enum_type;
  typedef std::vector< std::pair< std::string, T > > table_t;

  /// Create converter table.
  /**
   * @param table Conversion table to use.
   *
   */
  enum_converter( const table_t& table )
    : m_table( table )
  { }


  /// Convert from element name to value.
  /**
   * This method converts the supplied name into the associated value.
   *
   * @param name Name of the enum element.
   *
   * @return Enum value
   * @throws std::runtime_error if the string name is not in the table.
   */
  T from_string( const std::string& name )
  {
    for( const auto& elem : m_table )
    {
      if ( elem.first == name ) return elem.second;
    }

    std::stringstream str;
    str << "Unknown name for enum: \"" << name
        << "\". Valid names are: " << element_name_string();

    throw std::runtime_error( str.str() );
  }

  /// Convert from enum code to name string.
  /**
   * This method converts the supplied value to the associated string.
   *
   * @param val Enum value to convert
   *
   * @return String name of element
   * @throws std::runtime_error if the value is not in the table.
   */
  std::string to_string( T val )
  {
    for( const auto& elem : m_table )
    {
      if ( elem.second == val ) return elem.first;
    }

    throw std::runtime_error( "Could not convert enum value to string" );
  }

  /// Get list of available names for enum.
  /**
   * This method returns a single string with all available names for
   * the enum elements. Each element is surrounded with double quotes
   * and commas are added as appropriate.
   *
   * @return String of element names.
   */
  std::string element_name_string() const
  {
    bool first(true);
    std::stringstream str;

    for( const auto& elem : m_table )
    {
      if ( ! first )
      {
        str << ", ";
      }
      else
      {
        first = false;
      }

      str << "\"" << elem.first << "\"";
    }

    str <<".";
    return str.str();
  }

private:
  table_t  m_table;
};

} } // end namespace

/// Shorthand method of creating enum converter classes.
/**
 * This macro creates a enum converter class that is derived from the
 * base class and is initialized with the supplied conversion pairs.
 *
 * @param CN Converter class name
 * @param T Enum type
 * @param ... list of initialization pairs in the form { "name", value }, ...
 */
#define ENUM_CONVERTER(CN, T, ...)                      \
struct CN : kwiver::vital::enum_converter < T > {       \
  CN()                                                  \
      : enum_converter( {                               \
        __VA_ARGS__                                     \
          } ) {}                                        \
  typedef T enum_type;                                  \
};

#endif /* VITAL_UTIL_ENUM_CONVERTER_H */
