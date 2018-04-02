/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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

#ifndef VITAL_RANGE_DEFS_H
#define VITAL_RANGE_DEFS_H

#include <tuple>

namespace kwiver {
namespace vital {
namespace range {

/**
 * \file
 * \brief core types and macros for implementing range utilities.
 */

#define KWIVER_UNPACK_TOKENS(...) __VA_ARGS__

// ----------------------------------------------------------------------------
#define KWIVER_RANGE_ADAPTER_TEMPLATE( name, args, arg_names ) \
  template < KWIVER_UNPACK_TOKENS args > \
  struct name##_view_adapter_t \
  { \
    template < typename Range > \
    static name##_view< KWIVER_UNPACK_TOKENS arg_names, Range > \
    adapt( Range const& range ) \
    { return { range }; } \
  }; \
  \
  template < KWIVER_UNPACK_TOKENS args > \
  range_adapter_t< name##_view_adapter_t< KWIVER_UNPACK_TOKENS arg_names > > \
  name();

// ----------------------------------------------------------------------------
#define KWIVER_RANGE_ADAPTER_FUNCTION( name ) \
  template < typename... Args > \
  name##_view_adapter_t< Args... > \
  name( Args... args ) \
  { return { args... }; } \
  \
  template < typename Range, typename... Args > \
  auto \
  operator|( \
    Range const& range, \
    name##_view_adapter_t< Args... > const& adapter ) \
  -> decltype( adapter.adapt( range ) ) \
  { \
    return adapter.adapt( range ); \
  }

// ----------------------------------------------------------------------------
template < typename GenericAdapter >
struct range_adapter_t {};

// ----------------------------------------------------------------------------
template < typename Range, typename Adapter >
auto
operator|(
  Range const& range,
  range_adapter_t< Adapter >(*)() )
-> decltype( Adapter::adapt( range ) )
{
  return Adapter::adapt( range );
}

// ----------------------------------------------------------------------------
template < typename Functor >
struct function_detail : function_detail< decltype( &Functor::operator() ) >
{};

// ----------------------------------------------------------------------------
template < typename ReturnType, typename... ArgsType >
struct function_detail< ReturnType (&)( ArgsType... ) >
{
  using arg_type_t = std::tuple< ArgsType... >;
  using return_type_t = ReturnType;
};

// ----------------------------------------------------------------------------
template < typename ReturnType, typename... ArgsType >
struct function_detail< ReturnType (*)( ArgsType... ) >
{
  using arg_type_t = std::tuple< ArgsType... >;
  using return_type_t = ReturnType;
};

// ----------------------------------------------------------------------------
template < typename Object, typename ReturnType, typename... ArgsType >
struct function_detail< ReturnType ( Object::* )( ArgsType... ) const >
{
  using arg_type_t = std::tuple< ArgsType... >;
  using return_type_t = ReturnType;
};

} } } // end namespace

#endif
