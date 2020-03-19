// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief This file contains the interface for vital metadata.
 */

#ifndef KWIVER_VITAL_METADATA_H_
#define KWIVER_VITAL_METADATA_H_

#include <vital/vital_export.h>

#include <vital/any.h>
#include <vital/exceptions/metadata.h>
#include <vital/types/metadata_tags.h>
#include <vital/types/timestamp.h>

#include <iostream>
#include <map>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <vector>

namespace kwiver {
namespace vital {

template <vital_metadata_tag tag> struct vital_meta_trait;

// -----------------------------------------------------------------
/**
 * \brief Abstract base class for metadata items
 * This class is the abstract base class for a single metadata
 * item. This mainly provides the interface for the type specific
 * derived classes.
 *
 * All metadata items need a common base class so they can be managed
 * in a collection.
 */
class VITAL_EXPORT metadata_item
{
public:
  virtual ~metadata_item() = default;

  /**
   * \brief Test if metadata item is valid.
   * This method tests if this metadata item is valid.
   *
   * \return \c true if the item is valid, otherwise \c false.
   *
   * \sa metadata::find
   */
  bool is_valid() const;

  /// \copydoc is_valid
  operator bool() const { return this->is_valid(); }

  /**
   * \brief Get name of metadata item.
   *
   * This method returns the descriptive name for this metadata item.
   *
   * \return Descriptive name of this metadata entry.
   */
  std::string const& name() const;

  /**
   * \brief Get vital metadata tag.
   *
   * This method returns the vital metadata tag enum value.
   *
   * \return Metadata tag value.
   */
  vital_metadata_tag tag() const { return m_tag; };

    /**
   * \brief Get metadata data type.
   *
   * This method returns the type-info for this metadata item.
   *
   * \return Type info for metadata tag.
   */
  virtual std::type_info const& type() const = 0;

  /**
   * \brief Get actual data for metadata item.
   *
   * This method returns the actual raw data for this metadata item as
   * a "any" object. Non-standard data types must be handled through this
   * call.
   *
   * \return Data for metadata item.
   */
  kwiver::vital::any data() const;

  template< typename T>
  bool data(T& val) const
  {
    if (typeid(T) == m_data.type())
    {
      val = kwiver::vital::any_cast<T>( data() );
      return true;
    }
    else
    {
      return false; // could throw
    }
  }

  /**
   * \brief Get metadata value as double.
   *
   * This method returns the metadata item value as a double or throws
   * an exception if data is not a double.
   *
   * \return Data for metadata item as double.
   * \throws bad_any_cast if data type is not really a double.
   */
  double as_double() const;

  /**
   * \brief Does this entry contain double type data?
   * This method returns \b true if this metadata item contains a data
   * value with double type.
   *
   * \return \b true if data type is double, \b false otherwise/
   */
  bool has_double() const;

  /**
   * \brief Get metadata value as uint64.
   *
   * This method returns the metadata item value as a uint64 or throws
   * an exception if data is not a uint64.
   *
   * \return Data for metadata item as uint64.
   * \throws bad_any_cast if data type is not really a uint64.
   */
  uint64_t as_uint64() const;

  /**
   * \brief Does this entry contain uint64 type data?
   *
   * This method returns \b true if this metadata item contains a data
   * value with uint64 type.
   *
   * \return \b true if data type is uint64, \b false otherwise/
   */
  bool has_uint64() const;

  /**
   * \brief Get metadata value as string.
   *
   * This method returns the metadata item value as a string.  If the
   * data is actually a string (as indicated by has_string() method)
   * the native value is returned. If the data is of another type, it
   * is converted to a string.
   *
   * \return Data for metadata item as string.
   */
  virtual std::string as_string() const = 0;

  /**
   * \brief Determine if item has string representation.
   *
   * This method returns \b true if this metadata item contains a data
   * value with std::string type.
   *
   * \return \b true if data type is std::string, \b false otherwise/
   */
  bool has_string() const;

  /**
   * \brief Print the value of this item to an output stream.
   *
   * This method is has the advantage over \c as_string() that it allow
   * control over string formatting (e.g. precision of floats).
   */
  virtual std::ostream& print_value(std::ostream& os) const = 0;

protected:
  metadata_item( std::string const& name,
                 kwiver::vital::any const& data,
                 vital_metadata_tag tag );
  metadata_item( std::string const& name,
                 kwiver::vital::any&& data,
                 vital_metadata_tag tag );

  const std::string m_name;
  const kwiver::vital::any m_data;
  const vital_metadata_tag m_tag;

private:
  friend class metadata;
  virtual metadata_item* clone() const = 0;
}; // end class metadata_item

// -----------------------------------------------------------------
/*
 * This class is returned when find can not locate the requested tag.
 *
 */
  class unknown_metadata_item
    : public metadata_item
  {
  public:
    // -- CONSTRUCTORS --
    unknown_metadata_item()
      : metadata_item( "Requested metadata item is not in collection", 0, VITAL_META_UNKNOWN )
    { }

    virtual bool is_valid() const { return false; }
    virtual vital_metadata_tag tag() const { return static_cast< vital_metadata_tag >(0); }
    virtual std::type_info const& type() const { return typeid( void ); }
    virtual std::string as_string() const { return "--Unknown metadata item--"; }
    virtual double as_double() const { return 0; }
    virtual double as_uint64() const { return 0; }
    virtual std::ostream& print_value(std::ostream& os) const
    {
      os << this->as_string();
      return os;
    }

  }; // end class unknown_metadata_item


// -----------------------------------------------------------------
/// Class for typed metadata values.
/**
 * \brief Class for typed metadata values.

 * This class represents a typed metadata item.
 *
 * NOTE: Does it really add any benefit to have the metadata item
 * object have a type in addition to the contained data type
 * (kwiver::vital::any)? The type from the traits could be used to
 * guide creating of the metadata item, but having this extra type
 * allows the contained and the assumed metadata type to be
 * different. How should we deal with that case?
 *
 * The advantage is that it is easier to convert to string if the data
 * type is known in advance rather than having to handle multiple
 * possibly unknown types at run time.
 *
 * \tparam TAG Metadata tag value
 * \tparam TYPE Metadata value representation type
 */
template<vital_metadata_tag TAG, typename TYPE>
class typed_metadata
  : public metadata_item
{
public:
  typed_metadata( std::string const& p_name, TYPE const& p_data )
    : metadata_item( p_name, p_data, TAG )
  {
  }

  typed_metadata( std::string const& p_name, TYPE&& p_data )
    : metadata_item( p_name, std::move( p_data ), TAG )
  {
  }

  typed_metadata( std::string const& p_name, kwiver::vital::any const& p_data )
    : metadata_item( p_name, p_data, TAG )
  {
    if ( p_data.type() != typeid(TYPE) )
    {
      std::stringstream msg;
      msg << "Creating typed_metadata object with data type ("
          << demangle( p_data.type().name() )
          << ") different from type object was created with ("
          << demangle( typeid(TYPE).name() ) << ")";
      VITAL_THROW( metadata_exception, msg.str() );
    }
  }

  virtual ~typed_metadata() = default;

  std::type_info const& type() const override { return typeid( TYPE ); }
  std::string as_string() const override
  {
    if ( this->has_string() )
    {
      return kwiver::vital::any_cast< std::string  > ( m_data );
    }

    // Else convert to a string
    const auto var = kwiver::vital::any_cast< TYPE > ( m_data );
    std::stringstream ss;

    ss << var;
    return ss.str();
  }

  // Print the value of this item to an output stream
  std::ostream& print_value(std::ostream& os) const override
  {
    TYPE var = kwiver::vital::any_cast< TYPE > ( m_data );
    os << var;
    return os;
  }

private:
  // Create a copy of this metadata item. This is needed to disconnect
  // a metadata item from a shared pointer.
  metadata_item* clone() const override
  {
    return new typed_metadata<TAG, TYPE>( *this );
  }
}; // end class typed_metadata

// -----------------------------------------------------------------
/**
 * \brief Collection of metadata.
 *
 * This class represents a set of metadata items.
 *
 * The concept is to provide a canonical set of useful metadata
 * entries that can be derived from a variety of sources.  Sources may
 * include KLV video metadata (e.g. 0104 and 0601 standards), image
 * file header data (e.g. EXIF), telemetry data from a robot, etc.
 * The original intent was that this metadata is associated with
 * either an image or video frame, but it could be used in other
 * contexts as well.
 *
 * Metadata items from the different sources are converted into a
 * small set of data types to simplify using these elements. Since the
 * data item is represented as a kwiver::vital::any object, the actual
 * type of the data contained is difficult to deal with if it is not
 * constrained. There are three data types that are highly recommended
 * for representing metadata. These types are:
 *
 * - double
 * - uint64
 * - std::string
 *
 * These data types are directly supported by the metadata_item
 * API. There are some exceptions to this guideline however. Generally
 * useful compound data items, such as lat/lon coordinates and image
 * corner points, are represented using standard vital data types to
 * make dealing with the data items easier. For example, if you want
 * corner points, they can be retrieved with one call rather than
 * doing eight calls and storing the values in some structure.
 *
 * Metadata items with integral values that are less than 64 bits will
 * be stored in a uint64 data type. The original data type can be
 * retrieved using static_cast<>().
 *
 * There may be cases where application specific data types are
 * required and these will have to be handled on an individual
 * basis. In this case, the metadata item will have to be queried
 * directly about its type and the data will have to be retrieved from
 * the \c any object carefully.
 *
 */
class VITAL_EXPORT metadata
{
public:
// The design for this collection requires that the elements in the
// collection are owned by the collection and are only returned by
// value. This is why the std::unique_ptr is used. Unfortunately, not
// all stdc implementations support maps with unique_ptrs. The
// following is done to work around this limitation.
#ifdef VITAL_STD_MAP_UNIQUE_PTR_ALLOWED
  using item_ptr = std::unique_ptr< metadata_item >;
#else
  using item_ptr = std::shared_ptr< metadata_item >;
#endif
  using metadata_map_t = std::map< vital_metadata_tag, item_ptr >;
  using const_iterator_t = metadata_map_t::const_iterator;

  metadata();
  ~metadata() = default;

  /**
   * \brief Add metadata item to collection.
   *
   * This method adds a metadata item to the collection. The collection takes
   * ownership of the \p item.
   *
   * \param item New metadata item to be added to the collection.
   */
  void add( std::unique_ptr< metadata_item >&& item );

  /**
   * \brief Add metadata item to collection.
   *
   * This method adds a metadata item to the collection. The collection makes
   * a copy of the item.
   *
   * \param item New metadata item to be copied into the collection.
   */
  void add_copy( std::shared_ptr<metadata_item const> const& item );

  //@{
  /**
   * \brief  Add metadata item to collection.
   *
   * This method creates a new metadata item and adds it to the collection.
   *
   * \tparam Tag Metadata tag value.
   * \param data Metadata value.
   */
  template < vital_metadata_tag Tag >
  void add( typename vital_meta_trait< Tag >::type&& data )
  {
    using tag_t = vital_meta_trait< Tag >;
    using result_t = typed_metadata< Tag, typename tag_t::type >;
    using ptr_t = std::unique_ptr< result_t >;
    this->add( ptr_t{ new result_t{ tag_t::name(), std::move( data ) } } );
  }

  template < vital_metadata_tag Tag >
  void add( typename vital_meta_trait< Tag >::type const& data )
  {
    using tag_t = vital_meta_trait< Tag >;
    using result_t = typed_metadata< Tag, typename tag_t::type >;
    using ptr_t = std::unique_ptr< result_t >;
    this->add( ptr_t{ new result_t{ tag_t::name(), data } } );
  }
  //@}

  /**
   * \brief Add metadata item to collection.
   *
   * This method creates a new metadata item and adds it to the collection.
   * Unlike ::add, this method must be used if the \p data is an ::any, as
   * overload resolution will fail otherwise.
   *
   * \tparam Tag Metadata tag value.
   * \param data Metadata value.
   */
  template < vital_metadata_tag Tag >
  void add_any( any const& data )
  {
    using tag_t = vital_meta_trait< Tag >;
    using result_t = typed_metadata< Tag, typename tag_t::type >;
    using ptr_t = std::unique_ptr< result_t >;
    this->add( ptr_t{ new result_t{ tag_t::name(), data } } );
  }

  /**
   * \brief Remove metadata item.
   *
   * The metadata item that corresponds with the tag is deleted if it is in the
   * collection.
   *
   * \param tag Tag of metadata to delete.
   *
   * \return \b true if specified item was found and deleted.
   */
  bool erase( vital_metadata_tag tag );

  /**
   * \brief  Determine if metadata collection has tag.
   *
   * This method determines if the specified tag is in this metadata
   * collection.
   *
   * \param tag Check for the presence of this tag.
   *
   * \return \b true if tag is in metadata collection, \b false otherwise.
   */
  bool has( vital_metadata_tag tag ) const;

  /**
   * \brief Find metadata entry for specified tag.
   *
   * This method looks for the metadata entrty corresponding to the supplied
   * tag. If the tag is not present in the collection, the result will be a
   * instance for which metadata_item::is_valid returns \c false and whose
   * behavior otherwise is unspecified.
   *
   * \param tag Look for this tag in collection of metadata.
   *
   * \return metadata item object for tag.
   */
  metadata_item const& find( vital_metadata_tag tag ) const;

  //@{
  /**
   * \brief Get starting iterator for collection of metadata items.
   *
   * This method returns the const iterator to the first element in
   * the collection of metadata items.
   *
   * Typical usage
   \code
   auto ix = metadata_collection->begin();
   vital_metadata_tag tag = ix->first;
   std::string name = ix->second->name();
   kwiver::vital::any data = ix->second->data();
   \endcode
   *
   * \return Iterator pointing to the first element in the collection.
   */
  const_iterator_t begin() const;
  const_iterator_t cbegin() const;
  //@}

  //@{
  /**
   * \brief Get ending iterator for collection of metadata.
   *
   * This method returns the ending iterator for the collection of
   * metadata items.
   *
   * Typical usage:
   \code
   auto eix = metadata_collection.end();
   for ( auto ix = metadata_collection.begin(); ix != eix; ix++)
   {
     // process metada items
   }
   \endcode
   * \return Ending iterator for collection
   */
  const_iterator_t end() const;
  const_iterator_t cend() const;
  //@}

  /**
   * \brief Get the number of metadata items in the collection.
   *
   * This method returns the number of elements in the
   * collection. There will usually be at least one element which
   * defines the souce of the metadata items.
   *
   * \return Number of elements in the collection.
   */
  size_t size() const;

  /**
   * \brief Test whether collection is empty.
   *
   * This method returns whether the collection is empty
   * (i.e. size() == 0).  There will usually be at least
   * one element which defines
   * the souce of the metadata items.
   *
   * \return \b true if collection is empty
   */
  bool empty() const;

  /**
   * \brief Set timestamp for this metadata set.
   *
   * This method sets that time stamp for this metadata
   * collection. This time stamp can be used to relate this metada
   * back to other temporal data like a video image stream.
   *
   * \param ts Time stamp to add to this collection.
   */
  void set_timestamp( kwiver::vital::timestamp const& ts );

  /**
   * \brief Return timestamp associated with these metadata.
   *
   * This method returns the timestamp associated with this collection
   * of metadata. The value may not be meaningful if it has not
   * been set by set_timestamp().
   *
   * \return Timestamp value.
   */
  kwiver::vital::timestamp const& timestamp() const;

  static std::string format_string( std::string const& val );

private:
  metadata_map_t m_metadata_map;
  kwiver::vital::timestamp m_timestamp;

}; // end class metadata

using metadata_sptr = std::shared_ptr< metadata >;
using metadata_vector = std::vector< metadata_sptr >;

VITAL_EXPORT std::ostream& print_metadata( std::ostream& str, metadata const& metadata );
VITAL_EXPORT bool test_equal_content( const kwiver::vital::metadata& one,
                                      const kwiver::vital::metadata& other );

} } // end namespace

#endif
