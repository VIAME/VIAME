#include <vistk/pipeline/process.h>

using namespace vistk;

class compare_string_process
  : public process
{
  public:
    compare_string_process(config_t const& config);
    ~compare_string_process();

    //void _init();
    void _step();

    void _connect_input_port(port_t const& port, edge_ref_t const& edge);
    void _connect_output_port(port_t const& port, edge_ref_t const& edge);

    port_info_t _input_port_info(port_t const& port) const;
    port_info_t _output_port_info(port_t const& port) const;

    ports_t _input_ports() const;
    ports_t _output_ports() const;

    config::keys_t _available_config() const;
    conf_info_t _config_info(config::key_t const& key) const;
  private:
    class priv;
    boost::shared_ptr<priv> d;
};

class compare_string_process::priv
{
  public:
    priv(bool icase);
    ~priv();

    bool const ignore_case;

    edge_ref_t input_edge_str1;
    edge_ref_t input_edge_str2;

    edge_group_t output_edges;

    conf_info_t icase_conf_info;

    port_info_t str1_port_info;
    port_info_t str2_port_info;
    port_info_t output_port_info;

    static config::key_t const CONFIG_ICASE;
    static bool const DEFAULT_ICASE;
    static port_t const PORT_STRING1;
    static port_t const PORT_STRING2;
    static port_t const PORT_OUTPUT;
};

config::key_t const compare_string_process::priv::CONFIG_ICASE = config::key_t("ignore_case");
bool const compare_string_process::priv::DEFAULT_ICASE = false;
process::port_t const compare_string_process::priv::PORT_STRING1 = process::port_t("string1");
process::port_t const compare_string_process::priv::PORT_STRING2 = process::port_t("string2");
process::port_t const compare_string_process::priv::PORT_OUTPUT = process::port_t("are_same");

compare_string_process
::compare_string_process(config_t const& config)
  : process(config)
{
  bool const icase = config->get_value<bool>(priv::CONFIG_ICASE, priv::DEFAULT_ICASE);

  d = boost::shared_ptr<priv>(new priv(icase));
}

config::keys_t
compare_string_process
::_available_config() const
{
  config::keys_t keys = process::_available_config();

  keys.push_back(priv::CONFIG_ICASE);

  return keys;
}

process::conf_info_t
compare_string_process
::_config_info(config::key_t const& key) const
{
  if (key == priv::CONFIG_ICASE)
  {
     return d->icase_conf_info;
  }

  return process::_config_info(key);
}

void
compare_string_process
::_connect_input_port(port_t const& port, edge_ref_t edge)
{
  if (port == priv::PORT_STRING1)
  {
    if (d->input_edge_str1.use_count())
    {
      throw port_reconnect_exception(name(), port);
    }

    d->input_edge_str1 = edge;
  }
  else if (port == priv::PORT_STRING2)
  {
    if (d->input_edge_str2.use_count())
    {
      throw port_reconnect_exception(name(), port);
    }

    d->input_edge_str2 = edge;
  }
  else
  {
    process::_connect_input_port(port, edge);
  }
}

void
compare_string_process
::_connect_output_port(port_t const& port, edge_ref_t edge)
{
  if (port == priv::PORT_OUTPUT)
  {
    d->output_edges.push_back(edge);
  }
  else
  {
    process::_connect_output_port(port, edge);
  }
}

process::port_info_t
compare_string_process
::_input_port_info(port_t const& port) const
{
  if (port == priv::PORT_STRING1)
  {
    return d->str1_port_info;
  }
  else if (port == priv::PORT_STRING2)
  {
    return d->str2_port_info;
  }

  return process::_input_port_info(port);
}

process::port_info_t
compare_string_process
::_output_port_info(port_t const& port) const
{
  if (port == priv::PORT_OUTPUT)
  {
    return d->output_port_info;
  }

  return process::_output_port_info(port);
}

process::ports_t
compare_string_process
::_input_ports() const
{
  ports_t ports = process::_input_ports();

  ports.push_back(priv::PORT_STRING1);
  ports.push_back(priv::PORT_STRING2);

  return ports;
}

process::ports_t
compare_string_process
::_output_ports() const
{
  ports_t ports = process::_output_ports();

  ports.push_back(priv::PORT_OUTPUT);

  return ports;
}

compare_string_process::priv
::priv(bool icase)
  : ignore_case(icase)
{
  port_flags_t required;

  required.insert(flag_required);

  str1_port_info = port_info_t(new port_info(
    "string",
    required,
    port_description_t("The first string to compare.")));
  str2_port_info = port_info_t(new port_info(
    "string",
    required,
    port_description_t("The second string to compare.")));
  output_port_info = port_info_t(new port_info(
    "bool",
    required,
    port_description_t("Sends \'true\' if the strings were the same.")));

  icase_conf_info = conf_info_t(new conf_info(
    boost::lexical_cast<config::value_t>(priv::DEFAULT_ICASE),
    config::description_t("If set to \'true\', compares strings case insensitively.")));
}

#include <boost/algorithm/string/predicate.hpp>

void
compare_string_process
::_step()
{
  datum_t dat;
  stamp_t st;

  edge_datum_t const str1_dat = grab_from_edge_ref(d->input_edge_str1);
  edge_datum_t const str2_dat = grab_from_edge_ref(d->input_edge_str2);

  st = str1_dat.get<1>();

  edge_data_t input_dats;

  input_dats.push_back(str1_dat);
  input_dats.push_back(str2_dat);

  data_info_t const info = data_info(input_dats);

  switch (info.max_status)
  {
    case datum::DATUM_DATA:
      if (!info.same_color)
      {
        st = heartbeat_stamp();

        dat = datum::error_datum("The input edges are not colored the same.");
      }
      else if (!info.in_sync)
      {
        st = heartbeat_stamp();

        dat = datum::error_datum("The input edges are not synchronized.");
      }
      else
      {
        std::string const str1 = str1_dat.get<0>()->get_datum<std::string>();
        std::string const str2 = str2_dat.get<0>()->get_datum<std::string>();

        bool cmp = (str1 == str2);

        if (!cmp && d->ignore_case)
        {
           cmp = boost::iequals(str1, str2);
        }

        dat = datum::new_datum(cmp);
      }
      break;
    case datum::DATUM_EMPTY:
      dat = datum::empty_datum();
      break;
    case datum::DATUM_COMPLETE:
      mark_as_complete();
      dat = datum::complete_datum();
      break;
    case datum::DATUM_ERROR:
      dat = datum::error_datum("Error on the input edges.");
      break;
    case datum::DATUM_INVALID:
    default:
      dat = datum::error_datum("Unrecognized datum type.");
      break;
  }

  edge_datum_t const edat = edge_datum_t(dat, st);

  push_to_edges(d->output_edges, edat);

  process::_step();
}

compare_string_process
::~compare_string_process()
{
}

compare_string_process::priv
::~priv()
{
}

process_t
create_compare_string_process(config_t const& config)
{
  return process_t(new compare_string_process(config));
}

void
register_processes()
{
  process_registry_t const registry = process_registry::self();

  registry->register_process("compare_string", "Compares strings", create_compare_string_process);
}
