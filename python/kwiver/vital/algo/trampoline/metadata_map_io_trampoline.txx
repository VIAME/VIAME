// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file metadata_map_io_trampoline.txx
///
/// \brief trampoline for overriding virtual functions for
///        \link kwiver::vital::algo::metadata_map_io \endlink

#ifndef METADATA_MAP_IO_TRAMPOLINE_TXX
#define METADATA_MAP_IO_TRAMPOLINE_TXX

#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <python/kwiver/vital/util/pybind11.h>
#include <vital/algo/metadata_map_io.h>
#include <vital/types/vector.h>

namespace kwiver {

namespace vital {

namespace python {

template < class algorithm_def_metadata_map_io_base =
             kwiver::vital::algorithm_def< kwiver::vital::algo::metadata_map_io > >
class algorithm_def_metadata_map_io_trampoline
  : public algorithm_trampoline< algorithm_def_metadata_map_io_base >
{
public:
  using algorithm_trampoline< algorithm_def_metadata_map_io_base >
  ::algorithm_trampoline;

  std::string
  type_name() const override
  {
    VITAL_PYBIND11_OVERLOAD(
      std::string,
      kwiver::vital::algorithm_def< kwiver::vital::algo::metadata_map_io >,
      type_name, );
  }
};

template < class metadata_map_io_base = kwiver::vital::algo::metadata_map_io >
class metadata_map_io_trampoline
  : public algorithm_def_metadata_map_io_trampoline< metadata_map_io_base >
{
public:
  using algorithm_def_metadata_map_io_trampoline< metadata_map_io_base >
  ::algorithm_def_metadata_map_io_trampoline;

  kwiver::vital::metadata_map_sptr
  load_( std::istream& fin, std::string const& filename ) const override
  {
    VITAL_PYBIND11_OVERLOAD_PURE(
      kwiver::vital::metadata_map_sptr,
      kwiver::vital::algo::metadata_map_io,
      load_,
      fin,
      filename );
  }

  void
  save_( std::ostream& fout, kwiver::vital::metadata_map_sptr data,
         std::string const& filename ) const override
  {
    VITAL_PYBIND11_OVERLOAD_PURE(
      void,
      kwiver::vital::algo::metadata_map_io,
      save_,
      fout,
      data,
      filename );
  }
};

} // namespace python

} // namespace vital

} // namespace kwiver

#endif
