// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/core_types/color.h>

#include <pybind11/pybind11.h>
#include "python_fold.h"

namespace py = pybind11;

VIAME_PYTHON_MODULE( color, m )
{
  py::class_< viame::rgb_color,
    std::shared_ptr< viame::rgb_color > >( m, "RGBColor" )
    .def( py::init<>() )
    .def(
      py::init(
        [](float r, float g,
           float b){
          return viame::rgb_color(
            uint8_t( r ),
            uint8_t( g ), uint8_t( b ) );
        } ),
      py::arg( "r" ) = 0, py::arg( "g" ) = 0, py::arg( "b" ) = 0 )
    .def_readwrite( "r", &viame::rgb_color::r )
    .def_readwrite( "g", &viame::rgb_color::g )
    .def_readwrite( "b", &viame::rgb_color::b )
    .def(
      "__eq__",
      [](viame::rgb_color self, viame::rgb_color other){
        return ( ( self.r == other.r ) && ( self.g == other.g ) &&
                 ( self.b == other.b ) );
      } )
    .def(
      "__ne__",
      [](viame::rgb_color self, viame::rgb_color other){
        return ( ( self.r != other.r ) || ( self.g != other.g ) ||
                 ( self.b != other.b ) );
      } )
    .def(
      "__repr__", [](viame::rgb_color self){
        return "RGBColor{" + std::to_string( self.r ) + ", " +
               std::to_string( self.g ) + ", " + std::to_string( self.b ) + "}";
      } )
    .def(
      "__getitem__", [](viame::rgb_color self, int idx){
        switch( idx )
        {
          case 0: return self.r;
          case 1: return self.g;
          case 2: return self.b;
        }
        throw pybind11::index_error( "RGB can't have an index greater than 2" );
      } )
  ;
}
