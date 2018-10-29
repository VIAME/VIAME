// MPark.Variant
//
// Copyright Michael Park, 2015-2017
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.md or copy at http://boost.org/LICENSE_1_0.txt)

#ifndef VITAL_INTERNAL_IN_PLACE_HPP
#define VITAL_INTERNAL_IN_PLACE_HPP

#include <cstddef>

namespace kwiver {
namespace vital {

  struct in_place_t { explicit in_place_t() = default; };

  template <std::size_t I>
  struct in_place_index_t { explicit in_place_index_t() = default; };

  template <typename T>
  struct in_place_type_t { explicit in_place_type_t() = default; };

#if 0 // c++14 feature
  constexpr in_place_t in_place{};

  template <std::size_t I> constexpr in_place_index_t<I> in_place_index{};

  template <typename T> constexpr in_place_type_t<T> in_place_type{};
#endif

} }  // namespace

#endif  // VITAL_INTERNAL_IN_PLACE_HPP
