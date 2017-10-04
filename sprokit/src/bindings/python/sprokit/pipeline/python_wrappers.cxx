/*ckwg +29
 * Copyright 2012-2017 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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

#include <vital/config/config_block.h>
#include <sprokit/pipeline/process.h>
#include <sprokit/pipeline/process_cluster.h>
#include <sprokit/pipeline/scheduler.h>
#include <sprokit/pipeline/stamp.h>

#include <sprokit/python/util/python_exceptions.h>
#include <sprokit/python/util/python_gil.h>

#include <pybind11/pybind11.h>

/**
 * \file process_cluster.cxx
 *
 * \brief Python bindings for \link sprokit::process_cluster\endlink.
 */

using namespace pybind11;

// pybind11 doesn't allow non-const pair references
class wrap_port_addr
{
  public:
    wrap_port_addr() {};
    wrap_port_addr(sprokit::process::port_addr_t const& port_addr) : process(port_addr.first), port(port_addr.second) {};

    sprokit::process::name_t process;
    sprokit::process::port_t port;

    sprokit::process::port_addr_t get_addr() {return sprokit::process::port_addr_t(process,port);};
};

// Publisher class to access virtual methods
class wrap_scheduler
  : public sprokit::scheduler
{
  public:
    using scheduler::scheduler;
    using scheduler::_start;
    using scheduler::_wait;
    using scheduler::_pause;
    using scheduler::_resume;
    using scheduler::_stop;
};

// Trampoline class to allow us to to use virtual methods
class scheduler_trampoline
  : public sprokit::scheduler
{
  public:
    scheduler_trampoline(sprokit::pipeline_t const& pipe, kwiver::vital::config_block_sptr const& config) : scheduler( pipe, config ) {};
    void _start() override;
    void _wait() override;
    void _pause() override;
    void _resume() override;
    void _stop() override;
};

void
scheduler_trampoline
::_start()
{
  PYBIND11_OVERLOAD_PURE(
    void,
    scheduler,
    _start,
  );
}

void
scheduler_trampoline
::_wait()
{
  PYBIND11_OVERLOAD_PURE(
    void,
    scheduler,
    _wait,
  );
}

void
scheduler_trampoline
::_pause()
{
  PYBIND11_OVERLOAD_PURE(
    void,
    scheduler,
    _pause,
  );
}

void
scheduler_trampoline
::_resume()
{
  PYBIND11_OVERLOAD_PURE(
    void,
    scheduler,
    _resume,
  );
}

void
scheduler_trampoline
::_stop()
{
  PYBIND11_OVERLOAD_PURE(
    void,
    scheduler,
    _stop,
  );
}

// Publicist class to access protected methods 
class wrap_process
  : public sprokit::process
{
  public:
    using process::process;

    using process::_configure;
    using process::_init;
    using process::_reset;
    using process::_flush;
    using process::_step;
    using process::_reconfigure; 
    using process::_properties;
    using process::_input_ports;
    using process::_output_ports;
    using process::_input_port_info;
    using process::_output_port_info;
    using process::_set_input_port_type;
    using process::_set_output_port_type;
    using process::_available_config;
    using process::_config_info;
};

// Trampoline class to allow us to use virtual methods
class process_trampoline
  : public sprokit::process
{
  public:
    process_trampoline(kwiver::vital::config_block_sptr const& config) : process(config) {};
    void _configure() override;
    void _init() override;
    void _reset() override;
    void _flush() override;
    void _step() override;
    void _reconfigure(kwiver::vital::config_block_sptr const& config) override;
    sprokit::process::properties_t _properties() const override;
    sprokit::process::ports_t _input_ports() const override;
    sprokit::process::ports_t _output_ports() const override;
    port_info_t _input_port_info(port_t const& port) override;
    port_info_t _output_port_info(port_t const& port) override;
    bool _set_input_port_type(port_t const& port, port_type_t const& new_type) override;
    bool _set_output_port_type(port_t const& port, port_type_t const& new_type) override;
    kwiver::vital::config_block_keys_t _available_config() const override;
    sprokit::process::conf_info_t _config_info(kwiver::vital::config_block_key_t const& key) override;
};

void
process_trampoline
::_configure()
{
  PYBIND11_OVERLOAD(
    void,
    process,
    _configure,
  );
}

void
process_trampoline
::_init()
{
  PYBIND11_OVERLOAD(
    void,
    process,
    _init,
  );
}

void
process_trampoline
::_reset()
{
  PYBIND11_OVERLOAD(
    void,
    process,
    _reset,
  );
}

void
process_trampoline
::_flush()
{
  PYBIND11_OVERLOAD(
    void,
    process,
    _flush,
  );
}

void
process_trampoline
::_step()
{
  PYBIND11_OVERLOAD(
    void,
    process,
    _step,
  );
}

void
process_trampoline
::_reconfigure(kwiver::vital::config_block_sptr const& config)
{
  PYBIND11_OVERLOAD(
    void,
    process,
    _reconfigure,
    config
  );
}

sprokit::process::properties_t
process_trampoline
::_properties() const
{
  PYBIND11_OVERLOAD(
    sprokit::process::properties_t,
    process,
    _properties,
  );
}

sprokit::process::ports_t
process_trampoline
::_input_ports() const
{
  PYBIND11_OVERLOAD(
    sprokit::process::ports_t,
    process,
    _input_ports,
  );
}

sprokit::process::ports_t
process_trampoline
::_output_ports() const
{
  PYBIND11_OVERLOAD(
    sprokit::process::ports_t,
    process,
    _output_ports,
  );
}

sprokit::process::port_info_t
process_trampoline
::_input_port_info(port_t const& port)
{
  PYBIND11_OVERLOAD(
    sprokit::process::port_info_t,
    process,
    _input_port_info,
    port
  );
}

sprokit::process::port_info_t
process_trampoline
::_output_port_info(port_t const& port)
{
  PYBIND11_OVERLOAD(
    sprokit::process::port_info_t,
    process,
    _output_port_info,
    port
  );
}

bool
process_trampoline
::_set_input_port_type(port_t const& port, port_type_t const& new_type)
{
  PYBIND11_OVERLOAD(
    bool,
    process,
    _set_input_port_type,
    port, new_type
  );
}

bool
process_trampoline
::_set_output_port_type(port_t const& port, port_type_t const& new_type)
{
  PYBIND11_OVERLOAD(
    bool,
    process,
    _set_output_port_type,
    port, new_type
  );
}

kwiver::vital::config_block_keys_t
process_trampoline
::_available_config() const
{
  PYBIND11_OVERLOAD(
    kwiver::vital::config_block_keys_t,
    process,
    _available_config,
  );
}

sprokit::process::conf_info_t
process_trampoline
::_config_info(kwiver::vital::config_block_key_t const& key)
{
  PYBIND11_OVERLOAD(
    sprokit::process::conf_info_t,
    process,
    _config_info,
    key
  );
}

// We need this so we can access protected class methods
class wrap_process_cluster
  : public sprokit::process_cluster
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

    wrap_stamp(sprokit::stamp_t st) {stamp_ptr = st;};

    sprokit::stamp_t stamp_ptr;
    sprokit::stamp_t get_stamp() const {return stamp_ptr;};

    bool stamp_eq(wrap_stamp const& other);
    bool stamp_lt(wrap_stamp const& other);
};

wrap_stamp
new_stamp(sprokit::stamp::increment_t const& increment)
{
  sprokit::stamp_t st = sprokit::stamp::new_stamp(increment);
  return wrap_stamp(st);
}

wrap_stamp
incremented_stamp(wrap_stamp const& st)
{
  sprokit::stamp_t st_inc = sprokit::stamp::incremented_stamp(st.get_stamp());
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
class wrap_edge_datum : public sprokit::edge_datum_t
{
  public:
    wrap_edge_datum() : sprokit::edge_datum_t() {}
    wrap_edge_datum(sprokit::datum dat, wrap_stamp st)
               : sprokit::edge_datum_t(
                 std::make_shared<sprokit::datum>(dat), st.get_stamp())
               {}

    sprokit::datum get_datum() {return *datum;}
    void set_datum(sprokit::datum const& dat) {datum = std::make_shared<sprokit::datum>(dat);}
    wrap_stamp get_stamp() {return wrap_stamp(stamp);}
    void set_stamp(wrap_stamp const& st) {stamp = st.get_stamp();}
};
