#include <sprokit/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

using namespace sprokit;

class compare_string_process
  : public process
{
  public:
    compare_string_process(config_t const& config);
    ~compare_string_process();

    void _configure();
    void _step();
  private:
    class priv;
    std::unique_ptr<priv> d;
};

class compare_string_process::priv
{
  public:
    priv(bool icase);
    ~priv();

    bool const ignore_case;

    static config::key_t const config_icase;
    static config::value_t const default_icase;
    static port_t const port_string1;
    static port_t const port_string2;
    static port_t const port_output;
};

config::key_t const compare_string_process::priv::config_icase = config::key_t("ignore_case");
config::value_t const compare_string_process::priv::default_icase = config::value_t("false");
process::port_t const compare_string_process::priv::port_string1 = port_t("string1");
process::port_t const compare_string_process::priv::port_string2 = port_t("string2");
process::port_t const compare_string_process::priv::port_output = port_t("are_same");

compare_string_process
::compare_string_process(config_t const& config)
  : process(config)
{
  declare_configuration_key(
    priv::config_icase,
    priv::default_icase,
    config::description_t("If set to \'true\', compares strings case insensitively."));

  port_flags_t required;

  required.insert(flag_required);

  declare_input_port(
    priv::port_string1,
    "string",
    required,
    port_description_t("The first string to compare."));
  declare_input_port(
    priv::port_string2,
    "string",
    required,
    port_description_t("The second string to compare."));
  declare_output_port(
    priv::port_output,
    "bool",
    required,
    port_description_t("Sends \'true\' if the strings were the same."));
}

compare_string_process::priv
::priv(bool icase)
  : ignore_case(icase)
{
}

void
compare_string_process
::_configure()
{
  // Configure the process.
  {
    bool const icase = config_value<bool>(priv::config_icase);

    d.reset(new priv(icase));
  }
}

#include <boost/algorithm/string/predicate.hpp>

void
compare_string_process
::_step()
{
  std::string const str1 = grab_from_port_as<std::string>(priv::port_string1);
  std::string const str2 = grab_from_port_as<std::string>(priv::port_string2);

  bool cmp = (str1 == str2);

  if (!cmp && d->ignore_case)
  {
    cmp = boost::iequals(str1, str2);
  }

  push_to_port_as<bool>(priv::port_output, cmp);

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

void
register_processes()
{
  static process_registry::module_t const module_name = process_registry::module_t("example_processes");

  process_registry_t const registry = process_registry::self();

  if (registry->is_process_module_loaded(module_name))
  {
    return;
  }

  registry->register_process("compare_string", "Compare strings", create_process<compare_string_process>);
}
