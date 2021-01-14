// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Base interface for simple buffer classes
 */

#ifndef KWIVER_VITAL_UTIL_BUFFER_H_
#define KWIVER_VITAL_UTIL_BUFFER_H_

namespace kwiver {
namespace vital {


/// Base interface for simple buffer classe
template< class Data >
class buffer
{
public:
  virtual ~buffer() { }

  /// The maximum number of elements that can be stored in the buffer.
  virtual size_t capacity() const = 0;

  /// \brief The number of elements currently in the buffer.
  virtual size_t size() const = 0;

  /// \brief Return the item \a offset away from the last item.
  ///
  /// An \a offset of 0 refers to the last item.
  ///
  /// It is an error to ask for an offset beyond the number of items
  /// currently in the buffer.  Use has_datum_at() or length() to
  /// verify.
  virtual const Data& datum_at( unsigned offset ) const = 0;

  /// \brief Check if there is an \a offset away from the last item.
  ///
  /// An \a offset of 0 refers to the last item.
  ///
  /// If <tt>has_datum_at(x)</tt> returns \c true, then
  /// <tt>has_datum_at(y)</tt> will also return \c true for all 0 \<=
  /// y \<= x.
  virtual bool has_datum_at( unsigned offset ) const = 0;

  virtual size_t offset_of( Data const& ) const = 0;
};


} // end namespace vital
} // end namespace kwiver


#endif // KWIVER_VITAL_UTIL_BUFFER_H_
