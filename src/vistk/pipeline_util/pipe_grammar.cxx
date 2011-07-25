/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "pipe_grammar.h"

#include "pipe_declaration_types.h"

#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/std_pair.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix.hpp>

#include <string>

/**
 * \file pipe_grammar.cxx
 *
 * \brief The implementation of parsing pipeline blocks from a string.
 */

using namespace boost::spirit;

BOOST_FUSION_ADAPT_STRUCT(
  vistk::config_key_t,
  (vistk::config::keys_t, key_path)
  (boost::optional<vistk::config_flags_t>, flags)
  (boost::optional<vistk::config_provider_t>, provider)
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
  vistk::input_map_t,
  (boost::optional<vistk::map_flags_t>, flags)
  (vistk::process::port_t, from)
  (vistk::process::port_addr_t, to)
)

BOOST_FUSION_ADAPT_STRUCT(
  vistk::output_map_t,
  (boost::optional<vistk::map_flags_t>, flags)
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

namespace vistk
{

namespace
{

static token_t const config_block_name = token_t("config");
static token_t const process_block_name = token_t("process");
static token_t const connect_block_name = token_t("connect");
static token_t const group_block_name = token_t("group");
static token_t const map_block_name = token_t("map");

static token_t const from_name = token_t("from");
static token_t const to_name = token_t("to");
static token_t const type_token = token_t("::");

static token_t const flag_mapping_required_name = token_t("required");
static token_t const flag_config_readonly_name = token_t("ro");

static token_t const config_separator = token_t(config::block_sep);
static token_t const port_separator = token_t(".");
static token_t const flag_separator = token_t(",");
static token_t const flag_decl_open = token_t("[");
static token_t const flag_decl_close = token_t("]");
static token_t const provider_open = token_t("{");
static token_t const provider_close = token_t("}");

}

template<typename Iterator>
class VISTK_PIPELINE_UTIL_NO_EXPORT pipe_grammar
  : public qi::grammar<Iterator, pipe_blocks(), ascii::space_type>
{
  public:
    pipe_grammar();
    ~pipe_grammar();

    qi::rule<Iterator, token_t()> symbol;

    qi::rule<Iterator, map_flag_t(), ascii::space_type> map_flag;
    qi::rule<Iterator, map_flags_t(), ascii::space_type> map_flags;

    qi::rule<Iterator, config::key_t()> config_key_base;
    qi::rule<Iterator, config::keys_t(), qi::locals<config::keys_t>, ascii::space_type> config_key;
    qi::rule<Iterator, config_flag_t()> config_flag;
    qi::rule<Iterator, config_flags_t()> config_flags;
    qi::rule<Iterator, config_provider_t()> config_provider;
    qi::rule<Iterator, config::value_t(), ascii::space_type> config_value;
    qi::rule<Iterator, config_key_t(), ascii::space_type> config_key_full;
    qi::rule<Iterator, config_value_t(), ascii::space_type> partial_config_decl;
    qi::rule<Iterator, config_value_t(), ascii::space_type> config_decl;

    qi::rule<Iterator, process::name_t(), ascii::space_type> process_name;
    qi::rule<Iterator, process::port_t(), ascii::space_type> port_name;
    qi::rule<Iterator, process_registry::type_t(), ascii::space_type> type_name;
    qi::rule<Iterator, process_registry::type_t(), ascii::space_type> type_decl;
    qi::rule<Iterator, process::port_addr_t(), ascii::space_type> port_addr;

    qi::rule<Iterator, input_map_t(), ascii::space_type> input_map_block;
    qi::rule<Iterator, output_map_t(), ascii::space_type> output_map_block;

    qi::rule<Iterator, config_pipe_block(), ascii::space_type> config_block;
    qi::rule<Iterator, process_pipe_block(), ascii::space_type> process_block;
    qi::rule<Iterator, connect_pipe_block(), ascii::space_type> connect_block;
    qi::rule<Iterator, group_pipe_block(), qi::locals<config_values_t, input_maps_t, output_maps_t>, ascii::space_type> group_block;
    qi::rule<Iterator, pipe_blocks(), ascii::space_type> block_set;
};

pipe_blocks
parse_pipe_blocks_from_string(std::string const& str)
{
  static pipe_grammar<std::string::const_iterator> grammar;

  std::string::const_iterator i = str.begin();
  std::string::const_iterator const i_end = str.end();

  pipe_blocks blocks;

  bool const res = qi::phrase_parse(i, i_end, grammar, ascii::space, blocks);

  if (!res || (i != i_end))
  {
    /// \todo Throw an exception.
  }

  return blocks;
}

template<typename Iterator>
pipe_grammar<Iterator>
::pipe_grammar()
  : pipe_grammar::base_type(block_set, "pipe_grammar")
{
  symbol %=
    *(   ascii::alnum
     |   qi::lit('-')
     |   qi::lit('_')
     )
     -   qi::lit(config_separator);

  map_flag %=
     (  qi::lit(flag_mapping_required_name)
     );

  map_flags %=
     (   qi::lit(flag_decl_open)
     >> -(   map_flag
         >>  qi::lit(flag_separator)
         >>  map_flag
         )
     >>  qi::lit(flag_decl_close)
     );

  config_flag %=
     (  qi::lit(flag_config_readonly_name)
     );

  config_flags %=
     (   qi::lit(flag_decl_open)
     >> -(   config_flag
         >>  qi::lit(flag_separator)
         >>  config_flag
         )
     >>  qi::lit(flag_decl_close)
     );

  config_key_base %=
     (   symbol
     );

  config_key %=
     (   config_key_base[boost::phoenix::push_back(_a, _1)]
     >> *(   qi::lit(config_separator)
         >>  config_key_base[boost::phoenix::push_back(_a, _1)]
         )
     );

  config_provider %=
     (   qi::lit(provider_open)
     >>  symbol
     >>  qi::lit(provider_close)
     );

  config_key_full %=
     (   config_key
     >> -config_flags
     >> -config_provider
     );

  config_value %=
    *(   ascii::graph
     |   ascii::blank
     );

  config_decl %=
     (   config_key_full
     >>  config_value
     );

  partial_config_decl %=
     (   qi::lit(config_separator)
     >>  config_decl
     );

  config_block %=
     (   qi::lit(config_block_name)
     >>  config_key
     >> *partial_config_decl
     );

  process_name %=
     (   symbol
     );

  type_name %=
     (   symbol
     );

  type_decl %=
     (   qi::lit(type_token)
     >>  type_name
     );

  process_block %=
     (   qi::lit(process_block_name)
     >>  process_name
     >>  type_decl
     >> *partial_config_decl
     );

  port_addr %=
     (   process_name
     >>  qi::lit(port_separator)
     >>  port_name
     );

  connect_block %=
     (   qi::lit(connect_block_name)
     >>  qi::lit(from_name)
     >>  port_addr
     >>  qi::lit(to_name)
     >>  port_addr
     );

  input_map_block %=
     (   qi::lit(map_block_name)
     >> -map_flags
     >>  qi::lit(from_name)
     >>  port_name
     >>  qi::lit(to_name)
     >>  port_addr
     );

  output_map_block %=
     (   qi::lit(map_block_name)
     >> -map_flags
     >>  qi::lit(from_name)
     >>  port_addr
     >>  qi::lit(to_name)
     >>  port_name
     );

  group_block %=
     (   qi::lit(group_block_name)
     >>  process_name
     >> *(   partial_config_decl[boost::phoenix::push_back(_a, _1)]
         >>  input_map_block[boost::phoenix::push_back(_b, _1)]
         >>  output_map_block[boost::phoenix::push_back(_c, _1)]
         )
     );

  block_set %=
    *(   config_decl
     |   config_block
     |   process_block
     |   connect_block
     |   group_block
     );
}

template<typename Iterator>
pipe_grammar<Iterator>
::~pipe_grammar()
{
}

}
