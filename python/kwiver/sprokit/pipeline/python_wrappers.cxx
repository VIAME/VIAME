// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/config/config_block.h>
#include <sprokit/pipeline/process.h>
#include <sprokit/pipeline/process_cluster.h>
#include <sprokit/pipeline/scheduler.h>
#include <sprokit/pipeline/stamp.h>

#include <python/kwiver/vital/util/python_exceptions.h>

#include <pybind11/pybind11.h>

/**
 * \file python_wrappers.cxx
 *
 * \brief Python binding wrappers for some general purpose classes.
 */

using namespace pybind11;
namespace kwiver{
namespace sprokit{
namespace python{
// pybind11 doesn't allow non-const pair references
class wrap_port_addr
{
  public:
    wrap_port_addr() {};
    wrap_port_addr(::sprokit::process::port_addr_t const& port_addr) : process(port_addr.first), port(port_addr.second) {};

    ::sprokit::process::name_t process;
    ::sprokit::process::port_t port;

    ::sprokit::process::port_addr_t get_addr() {return ::sprokit::process::port_addr_t(process,port);};
};

// We need this so we can access protected class methods
class wrap_process_cluster
  : public ::sprokit::process_cluster
{
  public:

    using process_cluster::process_cluster;
    using process_cluster::map_config;
    using process_cluster::add_process;
    using process_cluster::map_input;
    using process_cluster::map_output;
    using process_cluster::connect;
    using process_cluster::_properties;
    using process_cluster::_reconfigure;
    using process::declare_input_port;
    using process_cluster::declare_output_port;
    using process_cluster::declare_configuration_key;
};

// We need to use this because PyBind11 has weird interactions with pointers
class wrap_stamp
{
  public:

    wrap_stamp(::sprokit::stamp_t st) {stamp_ptr = st;};

    ::sprokit::stamp_t stamp_ptr;
    ::sprokit::stamp_t get_stamp() const {return stamp_ptr;};

    bool stamp_eq(wrap_stamp const& other);
    bool stamp_lt(wrap_stamp const& other);
};

wrap_stamp
new_stamp(::sprokit::stamp::increment_t const& increment)
{
  ::sprokit::stamp_t st = ::sprokit::stamp::new_stamp(increment);
  return wrap_stamp(st);
}

wrap_stamp
incremented_stamp(wrap_stamp const& st)
{
  ::sprokit::stamp_t st_inc = ::sprokit::stamp::incremented_stamp(st.get_stamp());
  return wrap_stamp(st_inc);
}

bool
wrap_stamp
::stamp_eq(wrap_stamp const& other)
{
  return (*(get_stamp()) == *(other.get_stamp()));
}

bool
wrap_stamp
::stamp_lt(wrap_stamp const& other)
{
  return (*(get_stamp()) < *(other.get_stamp()));
}

// And because we're using wrap_stamp, we have to make it easier for edge_datum_t to access it
class wrap_edge_datum : public ::sprokit::edge_datum_t
{
  public:
    wrap_edge_datum() : ::sprokit::edge_datum_t() {}
    wrap_edge_datum(::sprokit::edge_datum_t dat)
               : ::sprokit::edge_datum_t(dat)
               {}
    wrap_edge_datum(::sprokit::datum dat, wrap_stamp st)
               : ::sprokit::edge_datum_t(
                 std::make_shared<::sprokit::datum>(dat), st.get_stamp())
               {}

    ::sprokit::datum get_datum() {return *datum;}
    void set_datum(::sprokit::datum const& dat) {datum = std::make_shared<::sprokit::datum>(dat);}
    wrap_stamp get_stamp() {return wrap_stamp(stamp);}
    void set_stamp(wrap_stamp const& st) {stamp = st.get_stamp();}
};
}
}
}
