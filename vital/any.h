/*ckwg +29
 * Copyright 2016-2017 by Kitware, Inc.
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

#ifndef KWIVER_VITAL_ANY_H
#define KWIVER_VITAL_ANY_H

#include <vital/vital_config.h>
#include <vital/util/demangle.h>

#include <algorithm>
#include <typeinfo>

namespace kwiver {
namespace vital {


// ----------------------------------------------------------------
/**
 * @brief Class that contains *any* data type.
 *
 * This class represents a single data item of indeterminate type.
 */
class any
{
public:
  /**
   * @brief Create empty object.
   *
   */
  any() noexcept
    : m_content( 0 )
  { }

  /**
   * @brief Create new object containing typed value.
   *
   * This CTOR creates a new object that holds the specified type and
   * value.
   *
   * @param value Data item (value and type) to be held in new object.
   */
  template < typename T >
  any( const T& value )
    : m_content( new internal_typed< T >( value ))
  { }

  /**
   * @brief Create new object form existing object.
   *
   * This copy CTOR creates a new object that contains the data value
   * and type of another any object.
   *
   * @param other Object to copy type and value from.
   */
  any( any const& other )
    : m_content( other.m_content ? other.m_content->clone() : 0 )
  { }

  ~any() noexcept
  {
    delete m_content;
  }


  // ------------------------------------------------------------------
  /**
   * @brief Swap value and type.
   *
   * This method swaps the specified value and type with this item.
   *
   * @param rhs Item to swap into this.
   *
   * @return Modified current (this) object.
   */
  any& swap(any& rhs) noexcept
  {
    std::swap(m_content, rhs.m_content);
    return *this;
  }

  // ------------------------------------------------------------------
  /**
   * @brief Assignment operator.
   *
   * This operator assigns the specified type and value to this object.
   *
   * @param rhs New value to assign to this object.
   *
   * @return Reference to this object.
   */
  template < typename T >
  any& operator=( T const& rhs )
  {
    any( rhs ).swap( *this );
    return *this;
  }

  // ------------------------------------------------------------------
  /**
   * @brief Assignment operator.
   *
   * This operator assigns the specified any object to this object.
   *
   * @param rhs New value to assign to this object.
   *
   * @return Reference to this object.
   */
  any& operator=( any rhs )
  {
    any( rhs ).swap( *this );
    return *this;
  }

  // ------------------------------------------------------------------
  /**
   * @brief Determine if this object has a value.
   *
   * This method returns \b true if this object has not been assigned
   * a value.
   *
   * @return \b true if no value in object, \b false if there is a
   * value.
   */
  bool empty() const noexcept
  {
    return ! m_content;
  }

  // ------------------------------------------------------------------
  /**
   * @brief Remove value from object.
   *
   * This method removes the current type and value from this
   * object. The empty() method will return /b true after this call.
   *
   */
  void clear() noexcept
  {
    any().swap( *this );
  }

  // ------------------------------------------------------------------
  /**
   * @brief Get typeid for current value.
   *
   * This method returns the std::type_info for the item contained in
   * this object. If this object is empty(), then the type info for \b
   * void is returned.
   *
   * You can get the type name string from the following, but the name
   * string may not be all that helpful.
   *
   \code
   kwiver::vital::any any_double(3.14159);
   std::cout << "Type name: " << any_double.type().name() << std::endl;
   \endcode
   *
   * @return The type info for the datum in this object is returned.
   */
  std::type_info const& type() const noexcept
  {
    return m_content ? m_content->type() : typeid(void);
  }


  /// Return demangled name of type contained in this object.
  /**
   * This method returns the demangled name of type contained in this
   * object.
   *
   * @return Demangled type name string.
   */
  std::string type_name() const noexcept
  {
    return demangle( this->type().name() );
  }

private:
  // ------------------------------------------------------------------
  // Base class for representing content
  class internal
  {
  public:
    virtual ~internal() { }
    virtual std::type_info const& type() const noexcept = 0;
    virtual internal* clone() const = 0;
  };

  // ------------------------------------------------------------------
  // type specific content
  template < typename T > class internal_typed : public internal
  {
  public:
    internal_typed( T const& value ) : m_any_data( value ) { }
    virtual std::type_info const& type() const noexcept
    {
      return typeid(T);
    }

    virtual internal* clone() const
    {
      return new internal_typed( m_any_data );
    }

    T m_any_data;

    // -- NOT IMPLEMENTED --
    internal_typed& operator=( const internal_typed& ) = delete;
  };


private:
  template < typename T >
  friend T* any_cast( any * aval ) noexcept;

  template < typename T >
  friend T any_cast(any const& aval);

  internal* m_content;
};


// ==================================================================
class  bad_any_cast : public std::bad_cast
{
public:

  /// Create bad cast exception;
  /**
   * This is the CTOR for the bnad any cast exception. A message is
   * created from the supplied mangled type names.
   *
   * @param from_type Mangled type name.
   * @param to_type Mangled type name.
   */
  bad_any_cast( std::string const& from_type,
                std::string const& to_type )
  {
    // Construct helpful message
    if( from_type != "")
    {
      m_message = "vital::bad_any_cast: failed conversion using kwiver::vital::any_cast from type \""
        + demangle( from_type ) + "\" to type \"" + demangle( to_type ) + "\"";
    }
    else
    {
      m_message = "vital::bad_any_cast: attempted to cast an uninitialized kwiver::vital::any object";
    }
  }

  virtual ~bad_any_cast() noexcept {}
  virtual const char * what() const noexcept
  {
    return m_message.c_str();
  }

private:
  std::string m_message;
};


// ==================================================================
// Casting functions
//
/// Get value from a container.
/**
 * This method returns a typed value from the any container. If the
 * conversion can not be completed, then an exception is thrown.
 *
 * @param aval Object that has the value.
 *
 * @return Value from object as specified type.
 */
template < typename T >
inline T*
any_cast( any* operand ) noexcept
{
  if ( operand && ( operand->type() == typeid( T ) ) )
  {
    return &static_cast< any::internal_typed< T >* > ( operand->m_content )->m_any_data;
  }

  return 0;
}


// ------------------------------------------------------------------
/// Get value from a container.
/**
 * This method returns a typed value from the any container. If the
 * conversion can not be completed, then an exception is thrown.
 *
 * @param aval Object that has the value.
 *
 * @return Value from object as specified type.
 */
template < typename T >
inline const T*
any_cast( any const* operand ) noexcept
{
  return any_cast< T > ( const_cast< any* > ( operand ) );
}


// ------------------------------------------------------------------
/// Get value from a container.
/**
 * This method returns a typed value from the any container. If the
 * conversion can not be completed, then an exception is thrown.
 *
 * @param aval Object that has the value.
 *
 * @return Value from object as specified type.
 */
template < typename T >
inline T
any_cast( any const& aval )
{
  // Is the type requested compatible with the type represented.
  if (aval.m_content)
  {
    if ( typeid( T ) == aval.m_content->type() )
    {
      return ( ( any::internal_typed< T >* )aval.m_content )->m_any_data;
    }

    throw bad_any_cast( aval.m_content->type().name(), typeid( T ).name() );
  }

  throw bad_any_cast( "", typeid( T ).name() );
}

} }  // end namespace

#endif /* KWIVER_VITAL_ANY_H */
