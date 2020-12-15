// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file \brief This file contains the any converter class
 */

#ifndef KWIVER_VITAL_UTIL_ANY_CONVERTER_H
#define KWIVER_VITAL_UTIL_ANY_CONVERTER_H

#include <vital/vital_config.h>
#include <vital/any.h>

#include <vector>
#include <memory>
#include <sstream>

namespace kwiver {
namespace vital {

namespace any_convert {

// ------------------------------------------------------------------
/*
 * Base class needed so converters can be stored in a collection.
 */
template < typename DEST >
struct convert_base
{
  convert_base() = default;
  virtual ~convert_base() = default;

  virtual bool can_convert( kwiver::vital::any const& data ) const = 0;
  virtual DEST convert( kwiver::vital::any const& data ) const = 0;
};

// ------------------------------------------------------------------
/*
 * Converter class that uses default static cast for conversion of
 * data types. This class can be specialized to create custom type
 * converters where it is not as simple as a static cast.
 */
template < typename DEST, typename SRC >
struct converter
  : public convert_base< DEST >
{
  converter() = default;
  virtual ~converter() = default;

  virtual bool can_convert( kwiver::vital::any const& data ) const override
  {
    return data.type() == typeid( SRC );
  }

  virtual DEST convert( kwiver::vital::any const& data ) const override
  {
    return static_cast< DEST > ( kwiver::vital::any_cast< SRC > ( data ) );
  }
};

// ------------------------------------------------------------------
template < typename SRC >
struct converter<std::string, SRC>
  : public convert_base< std::string >
{
  virtual bool can_convert( kwiver::vital::any const& data ) const override
  {
    return data.type() == typeid( SRC );
  }

  virtual std::string convert( kwiver::vital::any const& data ) const override
  {
    std::stringstream str;
    str << kwiver::vital::any_cast< SRC > ( data );
    return str.str();
  }
};

// ------------------------------------------------------------------
/**
 * This specialization is not strictly needed, but resolves compiler
 * warning C4800 (forcing value to bool) in Visual Studio.
 */
template < typename SRC >
struct converter<bool, SRC>
  : public convert_base< bool >
{
  virtual bool can_convert(kwiver::vital::any const& data) const override
  {
    return data.type() == typeid(SRC);
  }

  virtual bool convert(kwiver::vital::any const& data) const override
  {
    return kwiver::vital::any_cast< SRC > (data) != SRC(0);
  }
};

} // end namespace convert

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
any_to_int.add_converter<int>();      // self type needs to be added too

kwiver::vital::any some_data = ....;  // Get value in *any* object
int ival = any_to_int.convert( some_data ); // convert reasonable types to int
\endcode
 *
 * A more complicated conversion could be implemented for converting
 * various arbitrary values to a bool.
 *
\code
// make a custom specialization of converter struct
namespace kwiver {
namespace vital {
namespace any_convert {

template < >
struct converter< bool, std::string >
  : public convert_base< bool >
{
  converter()
  {
    // Add strings that can be converted to bool
    convert_map.insert( std::pair< std::string, bool > ( "yes", true ) );
    convert_map.insert( std::pair< std::string, bool > ( "YES", true ) );
    convert_map.insert( std::pair< std::string, bool > ( "no", false ) );
    convert_map.insert( std::pair< std::string, bool > ( "NO", false ) );
    convert_map.insert( std::pair< std::string, bool > ( "0", false ) );
    convert_map.insert( std::pair< std::string, bool > ( "zero", false ) );
    convert_map.insert( std::pair< std::string, bool > ( "1", true ) );
    convert_map.insert( std::pair< std::string, bool > ( "one", true ) );
    convert_map.insert( std::pair< std::string, bool > ( "on", true ) );
    convert_map.insert( std::pair< std::string, bool > ( "ON", true ) );
    convert_map.insert( std::pair< std::string, bool > ( "off", false ) );
    convert_map.insert( std::pair< std::string, bool > ( "OFF", false ) );
    convert_map.insert( std::pair< std::string, bool > ( "rabbit", true ) );
    convert_map.insert( std::pair< std::string, bool > ( "fish", false ) );
    convert_map.insert( std::pair< std::string, bool > ( "up", true ) );
    convert_map.insert( std::pair< std::string, bool > ( "down", false ) );
    convert_map.insert( std::pair< std::string, bool > ( "true", true ) );
    convert_map.insert( std::pair< std::string, bool > ( "false", false ) );
  }

  virtual ~converter() = default;

  virtual bool can_convert( kwiver::vital::any const & data ) const
  {
    return ( data.type() == typeid( std::string ) ) &&
           convert_map.find( kwiver::vital::any_cast< std::string > ( data ) ) != convert_map.end();
  }

  virtual bool convert( kwiver::vital::any const& data ) const
  {
    auto it = convert_map.find( kwiver::vital::any_cast< std::string > ( data ) );
    if ( it != convert_map.end() )
    {
      return it->second;
    }
    throw kwiver::vital::bad_any_cast( typeid( bool ).name(), typeid( std::string ).name() );
  }

private:
  std::map< std::string, bool > convert_map;
};

} } }     // end namespace

// Define the converter
kwiver::vital::any_converter< bool > convert_to_bool;

// Add converters for specific types
convert_to_bool.add_converter< bool > ();      // self type needs to be added too
convert_to_bool.add_converter< int > ();
convert_to_bool.add_converter< std::string > ();      // Use custom converter

std::string input;
std::getline( std::cin, input );
bool result;

if (convert_to_bool.can_convert( input ) )
{
  result = convert_to_bool.convert( input );
}
else
{
  std::cerr << "Can not convert \"" << input << "\" to bool\n";
}
\endcode
 *
 */
template <typename T>
class any_converter
{
public:

#ifdef VITAL_STD_MAP_UNIQUE_PTR_ALLOWED
  typedef std::unique_ptr< any_convert::convert_base< T > > converter_ptr;
#else
  typedef std::shared_ptr< any_convert::convert_base< T > > converter_ptr;
#endif

  any_converter() = default;
  virtual ~any_converter() = default;

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

  /// Add converter based on convert-from type.
  /**
   * Adds a new converter that handles a specific source type. The
   * types (DEST and SRC) must be convertable, either implicitly or
   * explicitly) for this to work
   *
   * tparam SRC Type from any to be converted.
   */
  template <typename SRC>
  void add_converter()
  {
    m_converter_list.push_back( converter_ptr( new any_convert::converter< T, SRC >() ) );
  }

  /// Add converter object.
  /**
   * Add a new converter object. The converter must be allocated from
   * the heap and ownership of the object is assumed by the converter.
   *
   * @param conv Converter object.
   */
  template<typename SRC>
  void add_converter( any_convert::convert_base< SRC >* conv )
  {
    m_converter_list.push_back( converter_ptr( conv ) );
  }

private:
  std::vector< converter_ptr > m_converter_list;

}; // end class any_converter

} } // end namespace

#endif // KWIVER_VITAL_UTIL_ANY_CONVERTER_H
