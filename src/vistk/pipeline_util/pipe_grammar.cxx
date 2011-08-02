/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "pipe_grammar.h"

#include "load_pipe_exception.h"

#include "pipe_declaration_types.h"

#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/std_pair.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix.hpp>
#include <boost/variant.hpp>

#include <string>
#include <sstream>
#include <ostream>

/**
 * \file pipe_grammar.cxx
 *
 * \brief The implementation of parsing pipeline blocks from a string.
 */

using namespace boost::spirit;

#ifndef DOXYGEN_IGNORE

BOOST_FUSION_ADAPT_STRUCT(
  vistk::config_key_options_t,
  (boost::optional<vistk::config_flags_t>, flags)
  (boost::optional<vistk::config_provider_t>, provider)
)

BOOST_FUSION_ADAPT_STRUCT(
  vistk::config_key_t,
  (vistk::config::keys_t, key_path)
  (vistk::config_key_options_t, options)
)

BOOST_FUSION_ADAPT_STRUCT(
  vistk::config_value_t,
  (vistk::config_key_t, key)
  (vistk::config::value_t, value)
)

BOOST_FUSION_ADAPT_STRUCT(
  vistk::config_pipe_block,
  (vistk::config::keys_t, key)
  (vistk::config_values_t, values)
)

BOOST_FUSION_ADAPT_STRUCT(
  vistk::process_pipe_block,
  (vistk::process::name_t, name)
  (vistk::process_registry::type_t, type)
  (vistk::config_values_t, config_values)
)

BOOST_FUSION_ADAPT_STRUCT(
  vistk::connect_pipe_block,
  (vistk::process::port_addr_t, from)
  (vistk::process::port_addr_t, to)
)

BOOST_FUSION_ADAPT_STRUCT(
  vistk::map_options_t,
  (boost::optional<vistk::process::port_flags_t>, flags)
)

BOOST_FUSION_ADAPT_STRUCT(
  vistk::input_map_t,
  (vistk::map_options_t, options)
  (vistk::process::port_t, from)
  (vistk::process::port_addr_t, to)
)

BOOST_FUSION_ADAPT_STRUCT(
  vistk::output_map_t,
  (vistk::map_options_t, options)
  (vistk::process::port_addr_t, from)
  (vistk::process::port_t, to)
)

BOOST_FUSION_ADAPT_STRUCT(
  vistk::group_pipe_block,
  (vistk::process::name_t, name)
  (vistk::config_values_t, config_values)
  (vistk::input_maps_t, input_mappings)
  (vistk::output_maps_t, output_mappings)
)

#endif

namespace vistk
{

namespace
{

static token_t const config_block_name = token_t("config");
static token_t const process_block_name = token_t("process");
static token_t const connect_block_name = token_t("connect");
static token_t const group_block_name = token_t("group");
static token_t const input_map_block_name = token_t("imap");
static token_t const output_map_block_name = token_t("omap");

static token_t const from_name = token_t("from");
static token_t const to_name = token_t("to");
static token_t const type_token = token_t("::");

static token_t const config_path_separator = token_t(config::block_sep);
static token_t const port_separator = token_t(".");
static token_t const flag_separator = token_t(",");
static token_t const flag_decl_open = token_t("[");
static token_t const flag_decl_close = token_t("]");
static token_t const provider_open = token_t("{");
static token_t const provider_close = token_t("}");

}

template<typename Iterator>
class VISTK_PIPELINE_UTIL_NO_EXPORT pipe_grammar
  : public qi::grammar<Iterator, pipe_blocks()>
{
  public:
    pipe_grammar();
    ~pipe_grammar();

    qi::rule<Iterator> opt_whitespace;
    qi::rule<Iterator> whitespace;
    qi::rule<Iterator> eol;
    qi::rule<Iterator> line_end;

    qi::rule<Iterator, config_flag_t()> config_flag;
    qi::rule<Iterator, config_flags_t()> config_flags;
    qi::rule<Iterator, config_flags_t()> config_flags_decl;

    qi::rule<Iterator, config_provider_t()> config_provider;
    qi::rule<Iterator, config_provider_t()> config_provider_decl;

    qi::rule<Iterator, config_key_options_t()> config_key_options;

    qi::rule<Iterator, config::key_t()> config_key;
    qi::rule<Iterator, config::keys_t()> config_key_path;
    qi::rule<Iterator, config::value_t()> config_value;

    qi::rule<Iterator, config_key_t()> config_key_full;

    qi::rule<Iterator, config_value_t()> partial_config_value_decl;
    qi::rule<Iterator, config_values_t()> partial_config_value_decls;

    qi::rule<Iterator, config_pipe_block()> config_block;

    qi::rule<Iterator, process_registry::type_t()> type_name;
    qi::rule<Iterator, process_registry::type_t()> type_decl;

    qi::rule<Iterator, process::name_t()> process_name;

    qi::rule<Iterator, process_pipe_block()> process_block;

    qi::rule<Iterator, process::port_t()> port_name;
    qi::rule<Iterator, process::port_addr_t()> port_addr;

    qi::rule<Iterator, connect_pipe_block()> connect_block;

    qi::rule<Iterator, process::port_flag_t()> map_flag;
    qi::rule<Iterator, process::port_flags_t(), qi::locals<process::port_flags_t> > map_flags;
    qi::rule<Iterator, process::port_flags_t()> map_flags_decl;

    qi::rule<Iterator, map_options_t()> map_options;

    qi::rule<Iterator, input_map_t()> input_map_block;
    qi::rule<Iterator, output_map_t()> output_map_block;

    qi::rule<Iterator, group_pipe_block(), qi::locals<config_values_t, input_maps_t, output_maps_t> > group_block;

    qi::rule<Iterator, pipe_blocks()> block_set;
};

class VISTK_PIPELINE_UTIL_NO_EXPORT printer
{
  public:
    typedef utf8_string string;

    printer(std::ostream& ostr);
    ~printer();

    void element(string const& tag, string const& value, int depth) const;
  private:
    static int const indent_width;

    std::ostream& m_ostr;
};

int const printer::indent_width = 4;

static void print_info(std::ostream& ostr, boost::spirit::info const& what);

pipe_blocks
parse_pipe_blocks_from_string(std::string const& str)
{
  static pipe_grammar<std::string::const_iterator> grammar;

  std::string::const_iterator i = str.begin();
  std::string::const_iterator const i_end = str.end();

  pipe_blocks blocks;

  try
  {
    qi::parse(i, i_end, grammar, blocks);
  }
  catch (qi::expectation_failure<std::string::const_iterator>& e)
  {
    std::ostringstream sstr;

    print_info(sstr, e.what_);

    throw failed_to_parse(sstr.str(), std::string(e.first, e.last));
  }

  if (i != i_end)
  {
    /// \todo Throw an exception.
  }

  return blocks;
}

template<typename Iterator>
pipe_grammar<Iterator>
::pipe_grammar()
  : pipe_grammar::base_type(block_set, "pipeline-declaration")
{
  opt_whitespace.name("opt-namespace");
  opt_whitespace %=
    *(  qi::blank
     );

  whitespace.name("whitespace");
  whitespace %=
    +(  qi::blank
     );

  eol.name("eol");
  eol %=
     (  qi::lit("\r\n")
     |  qi::lit("\n")
     );

  line_end.name("line-end");
  line_end %=
    +(  eol
     );

  config_flag.name("key-flag");
  config_flag %=
    +(  qi::alpha
     );

  config_flags.name("key-flags");
  config_flags %=
     (  config_flag
     %  qi::lit(flag_separator)
     );

  config_flags_decl.name("key-flags-decl");
  config_flags_decl %=
     (  qi::lit(flag_decl_open)
     >  config_flags
     >  qi::lit(flag_decl_close)
     );

  config_provider.name("key-provider");
  config_provider %=
    +(  qi::upper
     );

  config_provider_decl.name("key-provider-spec");
  config_provider_decl %=
     (  qi::lit(provider_open)
     >  config_provider
     >  qi::lit(provider_close)
     );

  config_key_options.name("key-options");
  config_key_options %=
     ( -config_flags_decl
     > -config_provider_decl
     );

  config_key.name("key-component");
  config_key %=
    +(  qi::alnum
     |  qi::lit('-')
     |  qi::lit('_')
     );

  config_key_path.name("key-path");
  config_key_path %=
     (  config_key
     %  qi::lit(config_path_separator)
     );

  config_value.name("key-value");
  config_value %=
    +(  qi::graph
     |  qi::lit(' ')
     |  qi::lit('\t')
     );

  config_key_full.name("full-key-path");
  config_key_full %=
     (  config_key_path
     >  config_key_options
     );

  partial_config_value_decl.name("partial-config-spec");
  partial_config_value_decl %=
     (  opt_whitespace
     >  qi::lit(config_path_separator)
     >  config_key_full
     >  whitespace
     >  config_value
     >  line_end
     );

  partial_config_value_decls.name("partial-configs-spec");
  partial_config_value_decls %=
    *(  partial_config_value_decl
     );

  config_block.name("config-block-spec");
  config_block %=
     (  opt_whitespace
     >  qi::lit(config_block_name)
     >  whitespace
     >  config_key_path
     >  line_end
     >  partial_config_value_decls
     );

  type_name.name("type-name");
  type_name %=
     (  config_key
     );

  type_decl.name("type-decl");
  type_decl %=
     (  qi::lit(type_token)
     >  whitespace
     >  type_name
     );

  process_name.name("port-process");
  process_name %=
     (  config_key
     );

  process_block.name("process-block-spec");
  process_block %=
     (  opt_whitespace
     >  qi::lit(process_block_name)
     >  whitespace
     >  process_name
     >  line_end
     >  opt_whitespace
     >  type_decl
     >  line_end
     >  partial_config_value_decls
     );

  port_name.name("port-name");
  port_name %=
     (  config_key
     );

  port_addr.name("port-addr");
  port_addr %=
     (  process_name
     >  qi::lit(port_separator)
     >  port_name
     );

  connect_block.name("connect-block-spec");
  connect_block %=
     (  opt_whitespace
     >  qi::lit(connect_block_name)
     >  whitespace
     >  qi::lit(from_name)
     >  whitespace
     >  port_addr
     >  line_end
     >  opt_whitespace
     >  qi::lit(to_name)
     >  whitespace
     >  port_addr
     >  line_end
     );

  map_flag.name("map-flag");
  map_flag %=
    +(  qi::alpha
     );

  map_flags.name("map-flags");
  map_flags %=
     (  map_flag[boost::phoenix::insert(_a, _1)]
     %  qi::lit(flag_separator)
     );

  map_flags_decl.name("map-flag-decl");
  map_flags_decl %=
     (  qi::lit(flag_decl_open)
     >  map_flags
     >  qi::lit(flag_decl_close)
     );

  map_options.name("map-options");
  map_options %=
     ( -map_flags_decl
     );

  input_map_block.name("input-mapping-spec");
  input_map_block %=
     (  opt_whitespace
     >  qi::lit(input_map_block_name)
     >  map_options
     >  whitespace
     >  qi::lit(from_name)
     >  whitespace
     >  port_name
     >  line_end
     >  opt_whitespace
     >  qi::lit(to_name)
     >  whitespace
     >  port_addr
     >  line_end
     );

  output_map_block.name("output-mapping-spec");
  output_map_block %=
     (  opt_whitespace
     >  qi::lit(output_map_block_name)
     >  map_options
     >  whitespace
     >  qi::lit(from_name)
     >  whitespace
     >  port_addr
     >  line_end
     >  opt_whitespace
     >  qi::lit(to_name)
     >  whitespace
     >  port_name
     >  line_end
     );

  group_block.name("group-block-spec");
  group_block %=
     (  opt_whitespace
     >  qi::lit(group_block_name)
     >  whitespace
     >  process_name
     >  line_end
     > *(  partial_config_value_decl[boost::phoenix::push_back(_a, _1)]
        |  input_map_block[boost::phoenix::push_back(_b, _1)]
        |  output_map_block[boost::phoenix::push_back(_c, _1)]
        )
     );

  block_set.name("block-spec");
  block_set %=
    *(  config_block
     |  process_block
     |  connect_block
     |  group_block
     );
}

template<typename Iterator>
pipe_grammar<Iterator>
::~pipe_grammar()
{
}

printer
::printer(std::ostream& ostr)
  : m_ostr(ostr)
{
}

printer
::~printer()
{
}

void
printer
::element(string const& tag, string const& value, int depth) const
{
  for (int i = 0; i < (depth * indent_width); ++i)
  {
    m_ostr << ' ';
  }

  m_ostr << "tag: " << tag;
  if (!value.empty())
  {
    m_ostr << ", value: " << value;
  }
}

void
print_info(std::ostream& ostr, boost::spirit::info const& what)
{
  printer pr(ostr);
  basic_info_walker<printer> walker(pr, what.tag, 0);
  boost::apply_visitor(walker, what.value);
}

}
