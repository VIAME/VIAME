/*ckwg +29
 * Copyright 2019 by Kitware, Inc.
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
  virtual unsigned capacity() const = 0;

  /// \brief The number of elements currently in the buffer.
  virtual unsigned size() const = 0;

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

  virtual unsigned offset_of( Data const& ) const = 0;
};


} // end namespace vital
} // end namespace kwiver


#endif // KWIVER_VITAL_UTIL_BUFFER_H_
