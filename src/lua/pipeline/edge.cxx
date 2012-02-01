/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/edge.h>
#include <vistk/pipeline/stamp.h>

#include <lua/helpers/lua_include.h>
#include <lua/helpers/lua_convert_vector.h>

#include <luabind/class.hpp>
#include <luabind/function.hpp>

/**
 * \file edge.cxx
 *
 * \brief Lua bindings for \link vistk::edge\endlink.
 */

extern "C"
{

int luaopen_vistk_pipeline_edge(lua_State* L);

}

using namespace luabind;

static vistk::datum_t edge_datum_datum(vistk::edge_datum_t const& edatum);
static void edge_datum_datum_set(vistk::edge_datum_t& edatum, vistk::datum_t const& dat);
static vistk::stamp_t edge_datum_stamp(vistk::edge_datum_t const& edatum);
static void edge_datum_stamp_set(vistk::edge_datum_t& edatum, vistk::stamp_t const& st);

int
luaopen_vistk_pipeline_edge(lua_State* L)
{
  open(L);

  module(L, "vistk")
  [
    namespace_("pipeline")
    [
      class_<vistk::edge_datum_t>("edge_datum")
        .def(constructor<vistk::datum_t, vistk::stamp_t>())
        .property("datum", &edge_datum_datum, &edge_datum_datum_set)
        .property("stamp", &edge_datum_stamp, &edge_datum_stamp_set)
    , class_<vistk::edge_data_t>("edge_data")
        .def(constructor<>())
    , class_<vistk::edges_t>("edges")
        .def(constructor<>())
    , class_<vistk::edge, vistk::edge_t>("edge")
        .def(constructor<vistk::config_t>())
        .def("makes_dependency", &vistk::edge::makes_dependency)
        .def("has_data", &vistk::edge::has_data)
        .def("full_of_data", &vistk::edge::full_of_data)
        .def("datum_count", &vistk::edge::datum_count)
        .def("push_datum", &vistk::edge::push_datum)
        .def("get_datum", &vistk::edge::get_datum)
        .def("peek_datum", &vistk::edge::peek_datum)
        .def("pop_datum", &vistk::edge::pop_datum)
        .def("set_upstream_process", &vistk::edge::set_upstream_process)
        .def("set_downstream_process", &vistk::edge::set_downstream_process)
        .def("mark_downstream_as_complete", &vistk::edge::mark_downstream_as_complete)
        .def("is_downstream_complete", &vistk::edge::is_downstream_complete)
    ]
  ];

  return 0;
}

vistk::datum_t
edge_datum_datum(vistk::edge_datum_t const& edatum)
{
  return edatum.get<0>();
}

void
edge_datum_datum_set(vistk::edge_datum_t& edatum, vistk::datum_t const& dat)
{
  boost::get<0>(edatum) = dat;
}

vistk::stamp_t
edge_datum_stamp(vistk::edge_datum_t const& edatum)
{
  return edatum.get<1>();
}

void
edge_datum_stamp_set(vistk::edge_datum_t& edatum, vistk::stamp_t const& st)
{
  boost::get<1>(edatum) = st;
}
