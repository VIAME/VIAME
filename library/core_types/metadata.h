// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief This file contains the interface for vital metadata.

#ifndef KWIVER_VITAL_METADATA_H_
#define KWIVER_VITAL_METADATA_H_

#include <viame/core_types/any.h>
#include <viame/algorithm_framework/exceptions/metadata.h>
#include <viame/core_types/geo_point.h>
#include <viame/core_types/geo_polygon.h>
#include <viame/core_types/metadata_tags.h>
#include <viame/core_types/metadata_traits.h>
#include <viame/core_types/rotation.h>
#include <viame/core_types/timestamp.h>
#include <viame/core_types/vital_types_export.h>
#include <viame/algorithm_framework/util/visit.h>

#include <iostream>
#include <map>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <variant>
#include <vector>

// needed to allow pybind11 wrap methods that return metadata_vector
#ifdef KWIVER_PYBIND11_INCLUDE
#include <pybind11/stl.h>
#endif

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
using metadata_value =
  std::variant<
    bool, int, uint64_t, double, std::string,
    geo_point, geo_polygon, rotation_d >;

// ----------------------------------------------------------------------------
/// Call \p visitor with template type parameter corresponding to \p type.
///
/// Simple wrapper around \c visit_variant_types specialized for the types of
/// \c metadata_value.
template < class Visitor >
void visit_metadata_types( Visitor&& visitor, std::type_info const& type );

// ----------------------------------------------------------------------------
/// Call \p visitor with template type parameter corresponding to \p type.
///
/// Simple wrapper around \c visit_variant_types_return specialized for the
/// types of \c metadata_value.
template < class ReturnT, class Visitor >
ReturnT visit_metadata_types_return(
  Visitor&& visitor,
  std::type_info const& type );

// ----------------------------------------------------------------------------
template < class Visitor >
void
visit_metadata_types( Visitor&& visitor, std::type_info const& type )
{
  visit_variant_types< metadata_value, Visitor >(
    std::forward< Visitor >( visitor ), type );
}

// ----------------------------------------------------------------------------
template < class ReturnT, class Visitor >
ReturnT
visit_metadata_types_return(
  Visitor&& visitor,
  std::type_info const& type )
{
  return visit_variant_types_return< ReturnT, metadata_value, Visitor >(
    std::forward< Visitor >( visitor ), type );
}

namespace metadata_detail {

// ----------------------------------------------------------------------------
template < class T > metadata_value
  convert_data( [[maybe_unused]] vital_metadata_tag tag, T const& data );

// ----------------------------------------------------------------------------
template < class T >
metadata_value
convert_data( [[maybe_unused]] vital_metadata_tag tag, T const& data )
{
  return data;
}

// ----------------------------------------------------------------------------
template <>
VITAL_TYPES_EXPORT
metadata_value
convert_data< any >( vital_metadata_tag tag, any const& data );

} // namespace metadata_detail

// ----------------------------------------------------------------------------
class VITAL_TYPES_EXPORT metadata_item
{
public:
  /// \throws logic_error If \p data's type does not match \p tag.
  template < class T >
  metadata_item( vital_metadata_tag tag, T&& data )
    : m_tag{ tag },
      m_data{ metadata_detail::convert_data( tag, std::forward< T >( data ) ) }
  {
    auto const& trait = tag_traits_by_tag( m_tag );
    if( trait.type() != this->type() )
    {
      std::stringstream ss;
      ss << "metadata_item constructed with tag " << trait.enum_name()
         << "expects type `" << trait.type_name() << "`; "
         << "received type `" << this->type_name() << "`";
      throw std::invalid_argument( ss.str() );
    }
  }

  bool operator==( metadata_item const& other ) const;
  bool operator!=( metadata_item const& other ) const;

  /// Test if the metadata item is valid.
  bool is_valid() const;

  /// \copydoc is_valid
  operator bool() const { return this->is_valid(); }

  /// Get the name of the metadata item.
  std::string name() const;

  /// Get the metadata item's tag.
  vital_metadata_tag
  tag() const { return m_tag; }

  /// Test if the metadata item has type \c T.
  template < class T >
  bool
  has() const
  {
    return this->type() == typeid( T );
  }

  /// Get the type of the metadata item's value.
  std::type_info const& type() const;

  /// Get the type name of the metadata item's value.
  std::string type_name() const;

  /// Get the value of this metadata item.
  metadata_value const& data() const;

  template < class T >
  T const&
  get() const
  {
    return std::get< T >( m_data );
  }

  /// Get the value of the metadata item as a \c double.
  ///
  /// \throws bad_any_cast If contained value is not a \c double.
  double as_double() const;

  /// Check if the metadata item contains a \c double value.
  bool has_double() const;

  /// Get the value of the metadata item as a \c uint64_t.
  ///
  /// \throws bad_any_cast If contained value is not a \c uint64_t.
  uint64_t as_uint64() const;

  /// Check if the metadata item contains a \c uint64_t value.
  bool has_uint64() const;

  /// Convert the value of the metadata item to \c std::string.
  ///
  /// \throws logic_error If the contained value is not a supported type.
  std::string as_string() const;

  /// Check if the metadata item contains a \c std::string value.
  bool has_string() const;

  /// Print the value of this item to an output stream.
  std::ostream& print_value( std::ostream& os ) const;

  /// Create a new copy of the metadata item.
  metadata_item* clone() const;

private:
  vital_metadata_tag m_tag;
  metadata_value m_data;
};

// ----------------------------------------------------------------------------
/// \brief Collection of metadata.
///
/// This class represents a set of metadata items.
///
/// The concept is to provide a canonical set of useful metadata
/// entries that can be derived from a variety of sources.  Sources may
/// include KLV video metadata (e.g. 0104 and 0601 standards), image
/// file header data (e.g. EXIF), telemetry data from a robot, etc.
/// The original intent was that this metadata is associated with
/// either an image or video frame, but it could be used in other
/// contexts as well.
///
/// Metadata items from the different sources are converted into a
/// small set of data types to simplify using these elements. Since the
/// data item is represented as a kwiver::vital::any object, the actual
/// type of the data contained is difficult to deal with if it is not
/// constrained. There are three data types that are highly recommended
/// for representing metadata. These types are:
///
/// - double
/// - uint64
/// - std::string
///
/// These data types are directly supported by the metadata_item
/// API. There are some exceptions to this guideline however. Generally
/// useful compound data items, such as lat/lon coordinates and image
/// corner points, are represented using standard vital data types to
/// make dealing with the data items easier. For example, if you want
/// corner points, they can be retrieved with one call rather than
/// doing eight calls and storing the values in some structure.
///
/// Metadata items with integral values that are less than 64 bits will
/// be stored in a uint64 data type. The original data type can be
/// retrieved using static_cast<>().
///
/// There may be cases where application specific data types are
/// required and these will have to be handled on an individual
/// basis. In this case, the metadata item will have to be queried
/// directly about its type and the data will have to be retrieved from
/// the \c any object carefully.
///
class VITAL_TYPES_EXPORT metadata
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
  metadata( metadata const& other );
  metadata( metadata&& other ) = default;
  virtual ~metadata() = default;
  metadata& operator=( metadata&& other ) = default;
  metadata& operator=( metadata const& other );

  bool operator==( metadata const& other ) const;
  bool operator!=( metadata const& other ) const;

  /// Create a deep copy of this object.
  virtual metadata* clone() const;

  /// \brief Add metadata item to collection.
  ///
  /// This method adds a metadata item to the collection. The collection takes
  /// ownership of the \p item.
  ///
  /// \param item New metadata item to be added to the collection.
  void add( std::unique_ptr< metadata_item >&& item );

  /// \brief Add metadata item to collection.
  ///
  /// This method adds a metadata item to the collection. The collection makes
  /// a copy of the item.
  ///
  /// \param item New metadata item to be copied into the collection.
  void add_copy( std::shared_ptr< metadata_item const > const& item );

  //@{
  /// \brief  Add metadata item to collection.
  ///
  /// This method creates a new metadata item and adds it to the collection.
  ///
  /// \tparam Tag Metadata tag value.
  /// \param data Metadata value.
  template < vital_metadata_tag Tag >
  void
  add( type_of_tag< Tag >&& data )
  {
    this->add(
      std::unique_ptr< metadata_item >(
        new metadata_item{ Tag,
                           std::move(
                             data ) } ) );
  }

  template < vital_metadata_tag Tag >
  void
  add( type_of_tag< Tag > const& data )
  {
    this->add(
      std::unique_ptr< metadata_item >(
        new metadata_item{ Tag,
                           data } ) );
  }

  //@}

  template < class T >
  void
  add( vital_metadata_tag tag, T&& data )
  {
    this->add(
      std::unique_ptr< metadata_item >(
        new metadata_item{ tag, std::forward< T >( data ) } ) );
  }

  void add_any( vital_metadata_tag tag, any const& data );

  /// \brief Add metadata item to collection.
  ///
  /// This method creates a new metadata item and adds it to the collection.
  /// Unlike ::add, this method must be used if the \p data is an ::any, as
  /// overload resolution will fail otherwise.
  ///
  /// \tparam Tag Metadata tag value.
  /// \param data Metadata value.
  template < vital_metadata_tag Tag >
  void
  add_any( any const& data )
  {
    this->add_any( Tag, data );
  }

  /// \brief Remove metadata item.
  ///
  /// The metadata item that corresponds with the tag is deleted if it is in the
  /// collection.
  ///
  /// \param tag Tag of metadata to delete.
  ///
  /// \return \b true if specified item was found and deleted.
  bool erase( vital_metadata_tag tag );

  /// \brief  Determine if metadata collection has tag.
  ///
  /// This method determines if the specified tag is in this metadata
  /// collection.
  ///
  /// \param tag Check for the presence of this tag.
  ///
  /// \return \b true if tag is in metadata collection, \b false otherwise.
  bool has( vital_metadata_tag tag ) const;

  /// \brief Find metadata entry for specified tag.
  ///
  /// This method looks for the metadata entrty corresponding to the supplied
  /// tag. If the tag is not present in the collection, the result will be a
  /// instance for which metadata_item::is_valid returns \c false and whose
  /// behavior otherwise is unspecified.
  ///
  /// \param tag Look for this tag in collection of metadata.
  ///
  /// \return metadata item object for tag.
  metadata_item const& find( vital_metadata_tag tag ) const;

  //@{
  /// \brief Get starting iterator for collection of metadata items.
  ///
  /// This method returns the const iterator to the first element in
  /// the collection of metadata items.
  ///
  /// Typical usage
  /// \code
  /// auto ix = metadata_collection->begin();
  /// vital_metadata_tag tag = ix->first;
  /// std::string name = ix->second->name();
  /// kwiver::vital::any data = ix->second->data();
  /// \endcode
  ///
  /// \return Iterator pointing to the first element in the collection.
  const_iterator_t begin() const;
  const_iterator_t cbegin() const;
  //@}

  //@{
  /// \brief Get ending iterator for collection of metadata.
  ///
  /// This method returns the ending iterator for the collection of
  /// metadata items.
  ///
  /// Typical usage:
  /// \code
  /// auto eix = metadata_collection.end();
  /// for ( auto ix = metadata_collection.begin(); ix != eix; ix++)
  /// {
  /// // process metada items
  /// }
  /// \endcode
  /// \return Ending iterator for collection
  const_iterator_t end() const;
  const_iterator_t cend() const;
  //@}

  /// \brief Get the number of metadata items in the collection.
  ///
  /// This method returns the number of elements in the
  /// collection. There will usually be at least one element which
  /// defines the souce of the metadata items.
  ///
  /// \return Number of elements in the collection.
  size_t size() const;

  /// \brief Test whether collection is empty.
  ///
  /// This method returns whether the collection is empty
  /// (i.e. size() == 0).  There will usually be at least
  /// one element which defines
  /// the souce of the metadata items.
  ///
  /// \return \b true if collection is empty
  bool empty() const;

  /// \brief Set timestamp for this metadata set.
  ///
  /// This method sets that time stamp for this metadata
  /// collection. This time stamp can be used to relate this metada
  /// back to other temporal data like a video image stream.
  ///
  /// \param ts Time stamp to add to this collection.
  void set_timestamp( kwiver::vital::timestamp const& ts );

  /// \brief Return timestamp associated with these metadata.
  ///
  /// This method returns the timestamp associated with this collection
  /// of metadata. The value may not be meaningful if it has not
  /// been set by set_timestamp().
  ///
  /// \return Timestamp value.
  kwiver::vital::timestamp timestamp() const;

  static std::string format_string( std::string const& val );

private:
  metadata_map_t m_metadata_map;
}; // end class metadata

using metadata_sptr = std::shared_ptr< metadata >;
using metadata_vector = std::vector< metadata_sptr >;

VITAL_TYPES_EXPORT std::ostream& print_metadata(
  std::ostream& str,
  metadata const& metadata );

} // namespace vital

}   // end namespace

#endif
