#include <vistk/pipeline/process.h>

#include <boost/make_shared.hpp>

using namespace vistk;

class compare_string_process
  : public process
{
  public:
    compare_string_process(config_t const& config);
    ~compare_string_process();

    //void _init();
    void _step();
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

    static config::key_t const config_icase;
    static bool const default_icase;
    static port_t const port_string1;
    static port_t const port_string2;
    static port_t const port_output;
};

config::key_t const compare_string_process::priv::config_icase = config::key_t("ignore_case");
bool const compare_string_process::priv::default_icase = false;
process::port_t const compare_string_process::priv::port_string1 = process::port_t("string1");
process::port_t const compare_string_process::priv::port_string2 = process::port_t("string2");
process::port_t const compare_string_process::priv::port_output = process::port_t("are_same");

compare_string_process
::compare_string_process(config_t const& config)
  : process(config)
{
  bool const icase = config->get_value<bool>(priv::config_icase, priv::default_icase);

  d = boost::make_shared<priv>(icase);

  declare_configuration_key(priv::config_icase, boost::make_shared<conf_info>(
    boost::lexical_cast<config::value_t>(priv::default_icase),
    config::description_t("If set to \'true\', compares strings case insensitively.")));

  port_flags_t required;

  required.insert(flag_required);

  declare_input_port(priv::port_string1, boost::make_shared<port_info>(
    "string",
    required,
    port_description_t("The first string to compare.")));
  declare_input_port(priv::port_string2, boost::make_shared<port_info>(
    "string",
    required,
    port_description_t("The second string to compare.")));
  declare_output_port(priv::port_output, boost::make_shared<port_info>(
    "bool",
    required,
    port_description_t("Sends \'true\' if the strings were the same.")));
}

compare_string_process::priv
::priv(bool icase)
  : ignore_case(icase)
{
}

#include <boost/algorithm/string/predicate.hpp>

void
compare_string_process
::_step()
{
  datum_t dat;
  stamp_t st;

  edge_datum_t const str1_dat = grab_from_port(priv::port_string1);
  edge_datum_t const str2_dat = grab_from_port(priv::port_string2);

  st = str1_dat.get<1>();

  edge_data_t input_dats;

  input_dats.push_back(str1_dat);
  input_dats.push_back(str2_dat);

  data_info_t const info = data_info(input_dats);

  switch (info->max_status)
  {
    case datum::DATUM_DATA:
      if (!info->same_color)
      {
        st = heartbeat_stamp();

        dat = datum::error_datum("The input edges are not colored the same.");
      }
      else if (!info->in_sync)
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

  push_to_port(priv::port_output, edat);

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
  return boost::make_shared<compare_string_process>(config);
}

void
register_processes()
{
  process_registry_t const registry = process_registry::self();

  registry->register_process("compare_string", "Compares strings", create_compare_string_process);
}
