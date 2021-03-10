// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_VITAL_ANY_H
#define KWIVER_VITAL_ANY_H

#include <vital/vital_config.h>
#include <vital/util/demangle.h>

#include <memory>
#include <typeinfo>
#include <type_traits>

#include <cstring>

namespace kwiver {

namespace vital {

// ============================================================================
/**
 * @brief Class that contains *any* data type.
 *
 * This class represents a single data item of indeterminate type.
 */
class any
{
  template < typename T >
  using is_self = std::is_same< typename std::decay< T >::type, any >;

  template < typename T >
  using non_self = typename std::enable_if< !is_self< T >::value >::type*;

public:
  /**
   * @brief Create empty object.
   *
   */
  any() noexcept
  {
  }

  /**
   * @brief Create new object containing typed value.
   *
   * This constructor creates a new object that holds the specified type and
   * value.
   *
   * @param value Data item (value and type) to be held in new object.
   */
  template < typename T, non_self< T > = nullptr >
  any( T&& value )
  {
    using value_t = typename std::decay< T >::type;
    this->m_content.reset(
      new internal_typed< value_t >{ std::forward< T >( value ) } );
  }

  /**
   * @brief Create new object from existing object.
   *
   * This copy constructor creates a new object that contains the data value
   * and type of another any object.
   *
   * @param other Object to copy type and value from.
   */
  any( any const& other )
    : m_content{ other.m_content ? other.m_content->clone() : nullptr }
  {
  }

  /**
   * @brief Create new object from an existing object.
   *
   * This move constructor creates a new object that contains the data value
   * and type of another any object. Afterwords, the other object is left in a
   * valid but unspecified state.
   *
   * @param other Object to move type and value from.
   */
  any( any&& other ) noexcept
  {
    this->swap( other );
  }

  ~any() noexcept
  {
  }

  /**
   * @brief Swap value and type.
   *
   * This method swaps the value and type of the specified any object with this
   * item.
   *
   * @param rhs Item to swap into this.
   *
   * @return Modified current (this) object.
   */
  any& swap( any& rhs ) noexcept
  {
    this->m_content.swap( rhs.m_content );
    return *this;
  }

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
  any& operator=( T&& rhs )
  {
    any{ std::forward< T >( rhs ) }.swap( *this );
    return *this;
  }

  /**
   * @brief Determine if this object has a value.
   *
   * This method returns \c true if this object has not been assigned a value.
   *
   * @return \c true if no value in object, \c false if there is a
   * value.
   */
  bool empty() const noexcept
  {
    return !m_content;
  }

  /**
   * @brief Remove value from object.
   *
   * This method removes the current type and value from this
   * object. The empty() method will return /b true after this call.
   *
   */
  void clear() noexcept
  {
    m_content.reset();
  }

  /**
   * @brief Get typeid for current value.
   *
   * This method returns the std::type_info for the item contained in this
   * object. If this object is empty(), the type info for \c void is returned.
   *
   * You can get the type name string from the following, but the name string
   * may not be all that helpful.
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

  /**
   * @brief Test type of current value.
   *
   * This method returns \c true if this object's value is of the type
   * specified by the template parameter.
   */
  template < typename T >
  bool is_type() const
  {
    if ( m_content )
    {
      auto const& my_type = this->m_content->type();
      return std::strcmp( typeid( T ).name(), my_type.name() ) == 0;
    }
    return std::is_same< T, void >::value;
  }

  /**
   * @brief Return demangled name of type contained in this object.
   *
   * This method returns the demangled name of type contained in this object.
   *
   * @return Demangled type name string.
   */
  std::string type_name() const noexcept
  {
    return demangle( this->type().name() );
  }

private:
  // --------------------------------------------------------------------------
  // Base class for representing content
  class internal
  {
  public:
    virtual ~internal() { }
    virtual std::type_info const& type() const noexcept = 0;
    virtual internal* clone() const = 0;
  };

  // --------------------------------------------------------------------------
  // Type specific content
  template < typename T > class internal_typed : public internal
  {
  public:
    internal_typed( T const& value ) : m_any_data( value ) {}
    internal_typed( T&& value ) : m_any_data( std::move( value ) ) {}

    virtual std::type_info const& type() const noexcept
    {
      return typeid(T);
    }

    virtual internal* clone() const
    {
      return new internal_typed{ m_any_data };
    }

    T m_any_data;

    // -- NOT IMPLEMENTED --
    internal_typed& operator=( const internal_typed& ) = delete;
  };

private:
  template < typename T >
  friend T* any_cast( any* ) noexcept;

  template < typename T >
  friend T const* any_cast( any const* ) noexcept;

  template < typename T >
  friend T any_cast( any const& );

  template < typename T >
  internal_typed< T >* content()
  {
    return static_cast< internal_typed< T >* >( this->m_content.get() );
  }

  template < typename T >
  internal_typed< T > const* content() const
  {
    return static_cast< internal_typed< T >* >( this->m_content.get() );
  }

  std::unique_ptr< internal > m_content;
};

// ============================================================================
class bad_any_cast : public std::bad_cast
{
public:

  /**
   * @brief Create bad cast exception;
   *
   * This is the constructor for the bad any cast exception. A message is
   * created from the supplied mangled type names.
   *
   * @param from_type Mangled type name.
   * @param to_type Mangled type name.
   */
  bad_any_cast( std::string const& from_type,
                std::string const& to_type )
  {
    // Construct helpful message
    if( !from_type.empty() )
    {
      m_message = "vital::bad_any_cast: failed conversion using kwiver::vital::any_cast from type \""
        + demangle( from_type ) + "\" to type \"" + demangle( to_type ) + "\"";
    }
    else
    {
      m_message = "vital::bad_any_cast: attempted to cast an uninitialized kwiver::vital::any object";
    }
  }

  virtual ~bad_any_cast() noexcept {};
  char const* what() const noexcept override
  {
    return m_message.c_str();
  }

private:
  std::string m_message;
};

///////////////////////////////////////////////////////////////////////////////

//BEGIN Casting functions

// ----------------------------------------------------------------------------
/**
 * @brief Get value pointer from a container.
 *
 * This method returns a typed pointer to the value from the ::any container.
 * If the types are incompatible, \c nullptr is returned.
 *
 * @param aval Object that has the value.
 *
 * @return Pointer to the value from the object as specified type,
 *         or \c nullptr if the conversion failed.
 */
template < typename T >
inline T*
any_cast( any* operand ) noexcept
{
  if ( operand && ( operand->is_type< T >() ) )
  {
    return &( operand->content< T >()->m_any_data );
  }
  return nullptr;
}

// ----------------------------------------------------------------------------
/**
 * @brief Get value pointer from a container.
 *
 * This method returns a typed pointer to the value from the ::any container.
 * If the types are incompatible, \c nullptr is returned.
 *
 * @param aval Object that has the value.
 *
 * @return Pointer to the value from the object as specified type,
 *         or \c nullptr if the conversion failed.
 */
template < typename T >
inline T const*
any_cast( any const* operand ) noexcept
{
  if ( operand && ( operand->is_type< T >() ) )
  {
    return &( operand->content< T >()->m_any_data );
  }
  return nullptr;
}

// ----------------------------------------------------------------------------
/**
 * @brief Get value from a container.
 *
 * This method returns a typed value from the any container. If the types are
 * incompatible, an exception is thrown.
 *
 * @param aval Object that has the value.
 *
 * @return Value from object as specified type.
 */
template < typename T >
inline T
any_cast( any const& aval )
{
  if ( aval.m_content )
  {
    if ( aval.is_type< T >() )
    {
      return aval.content< T >()->m_any_data;
    }

    throw bad_any_cast( aval.m_content->type().name(), typeid( T ).name() );
  }

  throw bad_any_cast( "", typeid( T ).name() );
}

} // namespace vital

} // namespace kwiver

//END Casting functions

#endif
