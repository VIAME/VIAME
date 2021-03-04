// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface of attribute_set class
 */

#ifndef KWIVER_VITAL_TTRIBUTE_SET_H
#define KWIVER_VITAL_TTRIBUTE_SET_H

#include <vital/vital_export.h>
#include <vital/vital_config.h>

#include <vital/any.h>
#include <vital/noncopyable.h>
#include <vital/exceptions/base.h>

#include <map>
#include <string>
#include <memory>

namespace kwiver {
namespace vital {

// ------------------------
class VITAL_EXPORT attribute_set_exception
  : public vital_exception
{
public:
  attribute_set_exception( std::string const& str );

  virtual ~attribute_set_exception() noexcept;
};

class attribute_set;
typedef std::shared_ptr< attribute_set > attribute_set_sptr;

// -----------------------------------------------------------------
/// General purpose attribute set.
/**
 * This class represents a set of general purpose attributes. The main
 * purpose of this class is to provide a mechanism for associating an
 * set of names attributes with another object.
 *
 * When integrating kwiver core capabilities into other systems, there
 * is usually some associated data that has to be transported with the
 * core data. This associated data is not used in the main algorithm
 * but must be available at the output so the artifact created can be
 * associated with the specific input.
 */
class VITAL_EXPORT attribute_set
  : private noncopyable
{
public:
  #ifdef VITAL_STD_MAP_UNIQUE_PTR_ALLOWED
  typedef std::unique_ptr< kwiver::vital::any > item_ptr;
#else
  typedef std::shared_ptr< kwiver::vital::any > item_ptr;
#endif
  typedef std::map< std::string, item_ptr > attribute_map_t;
  typedef attribute_map_t::const_iterator const_iterator_t;

  attribute_set();
  ~attribute_set();

  /**
   * @brief Create deep copy.
   *
   * This method creates a deep copy of this object.
   *
   * @return Managed copy of this object.
   */
  attribute_set_sptr clone() const;

  /**
   * @brief Add attribute top set.
   *
   * This method adds the attribute value wrapped in an 'any'
   * object. If the attribute already exists in the set, the existing
   * value will be replaced with the new one.
   *
   * @param name Name of the attribute
   * @param val Value of the attribute
   */
  void add( const std::string& name, const kwiver::vital::any& val );

  /**
   * @brief Add typed attribute to set.
   *
   * This method adds the typed attribute to the set.
   *
   * @param name Name if attribute
   * @param val Value of attribute
   */
  template<typename T>
  void add( const std::string& name, const T& val )
  {
    add( name, kwiver::vital::any( val ) );
  }

  /**
   * @brief Does the attribute exist in the set.
   *
   * This method determines if the named attribute exists in this set.
   *
   * @param name Name of the attribute
   *
   * @return \b true if the attribute is in the set.
   */
  bool has( const std::string& name ) const;

  /**
   * @brief Remove named attribute.
   *
   * This method removes the named attribute if is exists. Tf this set
   * does not have the named attribute, then no elements are removed.
   *
   * @param name Name of the attribute
   *
   * @return \b true if the attribute has been removed from this set;
   * \b false if the attribute is not in the set.
   */
  bool erase( const std::string& name );

  /**
   * @brief Get starting iterator for attribute set.
   *
   * This method returns the starting iterator for this set. This can
   * be used to iterate through all attributes in the set.
   *
   * @return Iterator pointing to the first attribute in the set.
   */
  const_iterator_t begin() const;

  /**
   * @brief Get ending iterator for attribute set.
   *
   * This method returns the ending iterator for this attribute
   * set.This can be used to iterate through all attributes in the
   * set.
   *
   * @return Ending iterator for this set.
   */
  const_iterator_t end() const;

  /**
   * @brief Return the number of attributes in the set.
   *
   * This method returns the number of attributes in the set.
   *
   * @return Number of attributes in the set.
   */
  size_t size() const;

  /**
   * @brief Is attribute set empty.
   *
   * This method determines if the attribute set is empty. \b true is
   * returned if the set is empty.
   *
   * @return \b true if the set is empty.
   */
  bool empty() const;

  /**
   * @brief Get raw data for attribute.
   *
   * this method returns the raw data item for the named attribute. If
   * the set does not contain the named item, an exception is thrown.
   *
   * In cases where the actual attribute type is known, the get()
   * method is preferred.
   *
   * @param name Name of the attribute
   *
   * @return Raw data for named attribute.
   * @throws attribute_set_exception if named attribute is not in the set.
   */
  kwiver::vital::any data( const std::string& name ) const;

  /**
   * @brief Get typed value from attribute set.
   *
   * This method returns the typed value of the attribute.
   *
   * @param name Name of attribute.
   *
   * @return Value of attribute.
   * @throws kwiver::vital::bad_any_cast if actual type does not match
   * requested type.
   */
  template<typename T>
  T get( const std::string& name ) const
  {
    return kwiver::vital::any_cast<T>( data(name ) );
  }

  /**
   * @brief Is the attribute of expected type.
   *
   * This method checks the actual attribute type against the template
   * type. \b true is returned if the type is as requested.
   *
   * @param name Name of the attribute
   *
   * @tparam T Expected type.
   *
   * @return \b true if the actual type is the same as the template
   * type.
   *
   * Example usage:
   \code
   if (is_type<std::string>("operator"))
   {
     auto val = get<std::string>("operator");
     ...
   }
   \endcode
   */
  template< typename T>
  bool is_type( const std::string& name ) const
  {
    const auto val = data( name );
    return (typeid(T) == val.type());
  }

private:
  attribute_map_t m_attr_map;
}; // end class attribute_set

typedef std::shared_ptr< attribute_set > attribute_set_sptr;

} } // end namespace

#endif /* KWIVER_VITAL_TTRIBUTE_SET_H */
