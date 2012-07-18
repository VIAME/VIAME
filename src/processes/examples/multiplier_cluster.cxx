/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "multiplier_cluster.h"

#include <vistk/pipeline/process_registry.h>

/**
 * \file multiplier_cluster.cxx
 *
 * \brief Implementation of the multiplier cluster.
 */

namespace vistk
{

class multiplier_cluster::priv
{
  public:
    priv();
    ~priv();

    void create_processes(multiplier_cluster const* q, process::name_t const& base_name);

    processes_t processes;
    connections_t input_mappings;
    connections_t output_mappings;
    connections_t internal_connections;

    static name_t const name_const;
    static name_t const name_multiplication;

    static type_t const type_const;
    static type_t const type_multiplication;

    static config::key_t const config_factor;
    static port_t const port_factor;
    static port_t const port_output;
};

process::name_t const multiplier_cluster::priv::name_const = name_t("/const");
process::name_t const multiplier_cluster::priv::name_multiplication = name_t("/multiplication");
process::type_t const multiplier_cluster::priv::type_const = type_t("const_number");
process::type_t const multiplier_cluster::priv::type_multiplication = type_t("multiplication");
config::key_t const multiplier_cluster::priv::config_factor = config::key_t("factor");
process::port_t const multiplier_cluster::priv::port_factor = port_t("factor");
process::port_t const multiplier_cluster::priv::port_output = port_t("product");

multiplier_cluster
::multiplier_cluster(config_t const& config)
  : process_cluster(config)
  , d(new priv)
{
  declare_configuration_key(
    priv::config_factor,
    config::value_t(),
    config::description_t("The value to start counting at."));

  port_flags_t required;

  required.insert(flag_required);

  declare_input_port(
    priv::port_factor,
    "integer",
    required,
    port_description_t("The factor to multiply."));
  declare_output_port(
    priv::port_output,
    "integer",
    required,
    port_description_t("Where the product will be available."));

  d->create_processes(this, name());
}

multiplier_cluster
::~multiplier_cluster()
{
}

vistk::processes_t
multiplier_cluster
::processes() const
{
  return d->processes;
}

vistk::process::connections_t
multiplier_cluster
::input_mappings() const
{
  return d->input_mappings;
}

vistk::process::connections_t
multiplier_cluster
::output_mappings() const
{
  return d->output_mappings;
}

vistk::process::connections_t
multiplier_cluster
::internal_connections() const
{
  return d->internal_connections;
}

multiplier_cluster::priv
::priv()
{
}

multiplier_cluster::priv
::~priv()
{
}

void
multiplier_cluster::priv
::create_processes(multiplier_cluster const* q, process::name_t const& base_name)
{
  name_t const full_name_const = base_name + name_const;
  name_t const full_name_mult = base_name + name_multiplication;

  process_registry_t const reg = process_registry::self();

  config_t const const_conf = config::empty_config();
  config_t const mult_conf = config::empty_config();

  const_conf->set_value("value", q->config_value<config::value_t>(config_factor));

  process_t const const_proc = reg->create_process(type_const, full_name_const, const_conf);
  process_t const mult_proc = reg->create_process(type_multiplication, full_name_mult, mult_conf);

  processes.push_back(const_proc);
  processes.push_back(mult_proc);

  input_mappings.push_back(connection_t(
        port_addr_t(base_name, port_factor),
        port_addr_t(full_name_mult, "factor1")));
  output_mappings.push_back(connection_t(
        port_addr_t(full_name_mult, "product"),
        port_addr_t(base_name, port_output)));
  internal_connections.push_back(connection_t(
        port_addr_t(full_name_const, "number"),
        port_addr_t(full_name_mult, "factor2")));
}

}
