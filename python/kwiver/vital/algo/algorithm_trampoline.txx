// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file algorithm_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions of vital::algorithm
 */

#ifndef ALGORITHM_TRAMPOLINE_TXX
#define ALGORITHM_TRAMPOLINE_TXX

#include <pybind11/pybind11.h>
#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/algorithm_framework/config/config_block.h>

namespace kwiver {

namespace vital {

namespace python {

template < class algorithm_base = kwiver::vital::algorithm >
class algorithm_trampoline : public algorithm_base
{
public:
  using algorithm_base::algorithm_base;

  kwiver::vital::config_block_sptr
  get_configuration() const override
  {
    PYBIND11_OVERLOAD(
      kwiver::vital::config_block_sptr,
      algorithm_base,
      get_configuration,
    );
  }

  void
  set_configuration( kwiver::vital::config_block_sptr config ) override
  {
    PYBIND11_OVERLOAD_PURE(
      void,
      algorithm_base,
      set_configuration,
      config
    );
  }

  bool
  check_configuration( kwiver::vital::config_block_sptr config ) const override
  {
    PYBIND11_OVERLOAD_PURE(
      bool,
      algorithm_base,
      check_configuration,
      config
    );
  }
};

} // namespace python

} // namespace vital

} // namespace kwiver

#endif
