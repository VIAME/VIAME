/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Python bindings for the OpenCV FileStorage subset reader
///
/// Small on purpose. They exist so that `tests/golden/calib` can replay its
/// `nodes` recording -- taken through `cv2.FileStorage`, which is the
/// definition of the format -- against the C++ reader that replaces it. A
/// recording nothing replays is a specification, not a test.
///
/// `tools/` reads calibration files with cv2 as well, and once phase 7 is
/// finished those can come here instead.

#include <viame/file_io/opencv_yaml.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

// ----------------------------------------------------------------------------
/// One node as the plain python value `cv2.FileStorage` would have given.
///
/// The shapes line up deliberately: a map is a dict, a sequence is a list,
/// and an `!!opencv-matrix` is the dict of its four fields rather than a
/// decoded array, because `dt` is what says whether the data is integers or
/// doubles and decoding here would throw it away.
py::object
to_python( viame::file_io::node const& value )
{
  using kind = viame::file_io::node::kind;

  switch( value.type() )
  {
    case kind::none:
      return py::none();

    case kind::integer:
      return py::int_( value.as_integer() );

    case kind::real:
      return py::float_( value.as_double() );

    case kind::string:
      return py::str( value.as_string() );

    case kind::sequence:
    {
      py::list out;
      for( auto const& element : value.values() )
      {
        out.append( to_python( element ) );
      }
      return std::move( out );
    }

    case kind::map:
    {
      py::dict out;
      for( auto const& entry : value.entries() )
      {
        out[ py::str( entry.first ) ] = to_python( entry.second );
      }
      return std::move( out );
    }
  }

  return py::none();
}

// ----------------------------------------------------------------------------
viame::file_io::node
from_python( py::handle value, bool matrix );

// ----------------------------------------------------------------------------
/// A dict, as a map -- or as a matrix if it has the four fields of one.
///
/// Guessing by shape rather than by a flag because the recording has no
/// flag: a document read back and written out again must produce the matrix
/// it started as, and `rows`, `cols`, `dt` and `data` together are what a
/// matrix is.
bool
looks_like_matrix( py::dict const& value )
{
  return value.contains( "rows" ) && value.contains( "cols" ) &&
         value.contains( "dt" ) && value.contains( "data" );
}

// ----------------------------------------------------------------------------
viame::file_io::node
from_python( py::handle value, bool matrix )
{
  if( value.is_none() )
  {
    return viame::file_io::node();
  }

  if( py::isinstance< py::bool_ >( value ) )
  {
    return viame::file_io::node(
      static_cast< long long >( value.cast< bool >() ? 1 : 0 ) );
  }

  if( py::isinstance< py::int_ >( value ) )
  {
    return viame::file_io::node( value.cast< long long >() );
  }

  if( py::isinstance< py::float_ >( value ) )
  {
    return viame::file_io::node( value.cast< double >() );
  }

  if( py::isinstance< py::str >( value ) )
  {
    return viame::file_io::node( value.cast< std::string >() );
  }

  if( py::isinstance< py::dict >( value ) )
  {
    auto const mapping = value.cast< py::dict >();
    std::vector< viame::file_io::node::entry > entries;

    for( auto const& item : mapping )
    {
      entries.emplace_back( item.first.cast< std::string >(),
                            from_python( item.second, false ) );
    }

    return viame::file_io::node::map_of( std::move( entries ),
                                         matrix || looks_like_matrix( mapping ) );
  }

  if( py::isinstance< py::list >( value ) ||
      py::isinstance< py::tuple >( value ) )
  {
    std::vector< viame::file_io::node > values;

    for( auto const& element : value )
    {
      values.push_back( from_python( element, false ) );
    }

    return viame::file_io::node::sequence_of( std::move( values ) );
  }

  throw py::type_error( "cannot write a value of this type to an OpenCV "
                        "document" );
}

} // namespace

// ----------------------------------------------------------------------------
PYBIND11_MODULE( _opencv_yaml, m )
{
  m.doc() =
    "VIAME's reader and writer for the OpenCV FileStorage subset "
    "(library/file_io/opencv_yaml), which replaced cv::FileStorage in "
    "P7-T05.";

  py::register_exception< viame::file_io::parse_error >(
    m, "ParseError", PyExc_ValueError );

  m.def(
    "read",
    []( std::string const& path ) {
      return to_python( viame::file_io::read( path ) );
    },
    py::arg( "path" ),
    "Parse an OpenCV YAML or XML document, choosing by its content. Returns "
    "the root as a dict; a map is a dict, a sequence is a list, and an "
    "opencv-matrix is the dict of its rows, cols, dt and data." );

  m.def(
    "to_yaml",
    []( py::dict const& document ) {
      return viame::file_io::to_yaml( from_python( document, false ) );
    },
    py::arg( "document" ),
    "The OpenCV YAML text for a document, laid out as FileStorage lays it "
    "out." );

  m.def(
    "write",
    []( std::string const& path, py::dict const& document ) {
      viame::file_io::write( path, from_python( document, false ) );
    },
    py::arg( "path" ), py::arg( "document" ),
    "Write a document as OpenCV YAML." );
}
