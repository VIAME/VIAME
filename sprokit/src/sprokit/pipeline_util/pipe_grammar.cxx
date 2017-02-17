/*ckwg +29
 * Copyright 2011-2015 by Kitware, Inc.
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

#include "pipe_grammar.h"

#include "load_pipe_exception.h"

#include "pipe_declaration_types.h"

#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/at.hpp>
#include <boost/fusion/include/std_pair.hpp>
#include <boost/spirit/include/qi.hpp>
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
  sprokit::config_key_options_t,
  (boost::optional<sprokit::config_flags_t>, flags)
  (boost::optional<sprokit::config_provider_t>, provider)
)

BOOST_FUSION_ADAPT_STRUCT(
  sprokit::config_key_t,
  (kwiver::vital::config_block_keys_t, key_path)
  (sprokit::config_key_options_t, options)
)

BOOST_FUSION_ADAPT_STRUCT(
  sprokit::config_value_t,
  (sprokit::config_key_t, key)
  (kwiver::vital::config_block_value_t, value)
)

BOOST_FUSION_ADAPT_STRUCT(
  sprokit::config_pipe_block,
  (kwiver::vital::config_block_keys_t, key)
  (sprokit::config_values_t, values)
)

BOOST_FUSION_ADAPT_STRUCT(
  sprokit::process_pipe_block,
  (sprokit::process::name_t, name)
  (sprokit::process::type_t, type)
  (sprokit::config_values_t, config_values)
)

BOOST_FUSION_ADAPT_STRUCT(
  sprokit::connect_pipe_block,
  (sprokit::process::port_addr_t, from)
  (sprokit::process::port_addr_t, to)
)

BOOST_FUSION_ADAPT_STRUCT(
  sprokit::cluster_config_t,
  (kwiver::vital::config_block_description_t, description)
  (sprokit::config_value_t, config_value)
)

BOOST_FUSION_ADAPT_STRUCT(
  sprokit::cluster_input_t,
  (sprokit::process::port_description_t, description)
  (sprokit::process::port_t, from)
  (sprokit::process::port_addrs_t, targets)
)

BOOST_FUSION_ADAPT_STRUCT(
  sprokit::cluster_output_t,
  (sprokit::process::port_description_t, description)
  (sprokit::process::port_addr_t, from)
  (sprokit::process::port_t, to)
)

BOOST_FUSION_ADAPT_STRUCT(
  sprokit::cluster_pipe_block,
  (sprokit::process::type_t, type)
  (sprokit::process::description_t, description)
  (sprokit::cluster_subblocks_t, subblocks)
)

#endif

namespace sprokit
{

namespace
{

static token_t const config_block_name = token_t("config");
static token_t const process_block_name = token_t("process");
static token_t const connect_block_name = token_t("connect");
static token_t const input_block_name = token_t("imap");
static token_t const output_block_name = token_t("omap");
static token_t const cluster_block_name = token_t("cluster");

static token_t const from_name = token_t("from");
static token_t const to_name = token_t("to");
static token_t const type_token = token_t("::");
static token_t const description_token = token_t(":#");

static token_t const config_path_separator = token_t(kwiver::vital::config_block::block_sep);
static token_t const port_separator = token_t(".");
static token_t const flag_separator = token_t(",");
static token_t const flag_decl_open = token_t("[");
static token_t const flag_decl_close = token_t("]");
static token_t const provider_open = token_t("{");
static token_t const provider_close = token_t("}");

}

template <typename Iterator>
class common_grammar
{
  public:
    common_grammar();
    ~common_grammar();

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

    qi::rule<Iterator, kwiver::vital::config_block_key_t()> decl_part;
    qi::rule<Iterator, kwiver::vital::config_block_key_t()> decl_component;
    qi::rule<Iterator, kwiver::vital::config_block_key_t()> config_key;
    qi::rule<Iterator, kwiver::vital::config_block_keys_t()> config_key_path;
    qi::rule<Iterator, kwiver::vital::config_block_value_t()> config_value;

    qi::rule<Iterator, config_key_t()> config_key_full;

    qi::rule<Iterator, config_value_t()> partial_config_value_decl;
    qi::rule<Iterator, config_values_t()> partial_config_value_decls;

    qi::rule<Iterator, config_pipe_block()> config_block;

    qi::rule<Iterator, process::type_t()> type_name;
    qi::rule<Iterator, process::type_t()> type_decl;

    qi::rule<Iterator, process::name_t()> process_name;

    qi::rule<Iterator, process_pipe_block()> process_block;

    qi::rule<Iterator, process::port_t()> port_name;
    qi::rule<Iterator, process::port_addr_t()> port_addr;

    qi::rule<Iterator, connect_pipe_block()> connect_block;
};

template <typename Iterator>
class pipe_grammar
  : public qi::grammar<Iterator, pipe_blocks()>
{
  public:
    pipe_grammar();
    ~pipe_grammar();
  private:
    common_grammar<Iterator> const common;

    qi::rule<Iterator, pipe_blocks()> pipe_block_set;
};

template <typename Iterator>
class cluster_grammar
  : public qi::grammar<Iterator, cluster_blocks()>
{
  public:
    cluster_grammar();
    ~cluster_grammar();
  private:
    common_grammar<Iterator> const common;

    qi::rule<Iterator, std::string()> description_decl;

    qi::rule<Iterator, cluster_config_t()> cluster_config_block;
    qi::rule<Iterator, process::port_addrs_t()> cluster_input_target;
    qi::rule<Iterator, cluster_input_t()> cluster_input_block;
    qi::rule<Iterator, cluster_output_t()> cluster_output_block;

    qi::rule<Iterator, cluster_pipe_block()> cluster_block;

    qi::rule<Iterator, cluster_blocks()> cluster_block_set;
};

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
  catch (qi::expectation_failure<std::string::const_iterator> const& e)
  {
    std::string::const_iterator const& begin = e.first;
    std::string::const_iterator const& end = e.last;
    std::ostringstream sstr;

    print_info(sstr, e.what_);

    throw failed_to_parse(sstr.str(), std::string(begin, end));
  }

  if (i != i_end)
  {
    throw failed_to_parse("End of file", std::string(i, i_end));
  }

  return blocks;
}

cluster_blocks
parse_cluster_blocks_from_string(std::string const& str)
{
  static cluster_grammar<std::string::const_iterator> grammar;

  std::string::const_iterator i = str.begin();
  std::string::const_iterator const i_end = str.end();

  cluster_blocks blocks;

  try
  {
    qi::parse(i, i_end, grammar, blocks);
  }
  catch (qi::expectation_failure<std::string::const_iterator> const& e)
  {
    std::string::const_iterator const& begin = e.first;
    std::string::const_iterator const& end = e.last;
    std::ostringstream sstr;

    print_info(sstr, e.what_);

    throw failed_to_parse(sstr.str(), std::string(begin, end));
  }

  if (i != i_end)
  {
    throw failed_to_parse("End of file", std::string(i, i_end));
  }

  return blocks;
}

template <typename Iterator>
common_grammar<Iterator>
::common_grammar()
  : opt_whitespace()
  , whitespace()
  , eol()
  , line_end()
  , config_flag()
  , config_flags()
  , config_flags_decl()
  , config_provider()
  , config_provider_decl()
  , config_key_options()
  , decl_part()
  , decl_component()
  , config_key()
  , config_key_path()
  , config_value()
  , config_key_full()
  , partial_config_value_decl()
  , partial_config_value_decls()
  , config_block()
  , type_name()
  , type_decl()
  , process_name()
  , process_block()
  , port_name()
  , port_addr()
  , connect_block()
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
    +(  qi::alnum
     |  qi::char_('-')
     |  qi::char_('_')
     |  qi::char_('=')
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

  decl_part.name("decl-part");
  decl_part %=
    +(  qi::alnum
     |  qi::char_('-')
     |  qi::char_('_')
     );

  // FIXME: there shouldn't be slashes next to each other. Basically,
  // decl_part % qi::char_('/') would be nice if it returned a single string
  // rather than a vector of decl_parts.
  decl_component.name("decl-component");
  decl_component %=
    +(  decl_part
     |  qi::char_('/')
     );

  config_key.name("key-component");
  config_key %=
    +(  qi::alnum
     |  qi::char_('-')
     |  qi::char_('_')
     |  qi::char_('/')
     |  qi::char_('.')
     );

  config_key_path.name("key-path");
  config_key_path %=
     (  config_key
     %  qi::lit(config_path_separator)
     );

  config_value.name("key-value");
  config_value %=
    +(  qi::graph
     |  qi::char_(' ')
     |  qi::char_('\t')
     );

  config_key_full.name("full-key-path");
  config_key_full %=
     (  config_key_path
     >  config_key_options
     );

  partial_config_value_decl.name("partial-config-spec");
  partial_config_value_decl %=
     (  opt_whitespace
     >> qi::lit(config_path_separator)
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
     >> qi::lit(config_block_name)
     >  whitespace
     >  config_key_path
     >  line_end
     >  partial_config_value_decls
     );

  type_name.name("type-name");
  type_name %=
     (  decl_part
     );

  type_decl.name("type-decl");
  type_decl %=
     (  qi::lit(type_token)
     >  whitespace
     >  type_name
     );

  process_name.name("port-process");
  process_name %=
     (  decl_component
     );

  process_block.name("process-block-spec");
  process_block %=
     (  opt_whitespace
     >> qi::lit(process_block_name)
     >  whitespace
     >  type_name
     >  line_end
     >  opt_whitespace
     >  type_decl
     >  line_end
     >  partial_config_value_decls
     );

  port_name.name("port-name");
  port_name %=
    +(  qi::alnum
     |  qi::char_('-')
     |  qi::char_('_')
     |  qi::char_('/')
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
     >> qi::lit(connect_block_name)
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
}

template <typename Iterator>
common_grammar<Iterator>
::~common_grammar()
{
}

template <typename Iterator>
pipe_grammar<Iterator>
::pipe_grammar()
  : pipe_grammar::base_type(pipe_block_set, "pipeline-declaration")
  , pipe_block_set()
{
  pipe_block_set.name("pipeline-blocks-spec");
  pipe_block_set %=
    *(  common.config_block
     |  common.process_block
     |  common.connect_block
     );
}

template <typename Iterator>
pipe_grammar<Iterator>
::~pipe_grammar()
{
}

template <typename Iterator>
cluster_grammar<Iterator>
::cluster_grammar()
  : cluster_grammar::base_type(cluster_block_set, "cluster-declaration")
  , description_decl()
  , cluster_config_block()
  , cluster_input_block()
  , cluster_output_block()
  , cluster_block()
  , cluster_block_set()
{
  description_decl.name("description-decl");
  description_decl %=
     (  qi::lit(description_token)
     >  common.whitespace
     >  common.config_value
     );

  cluster_config_block.name("cluster-config-spec");
  cluster_config_block %=
     (  common.opt_whitespace
     >> description_decl
     >> common.line_end
     >> common.partial_config_value_decl
     );

  cluster_input_target.name("cluster-input-target");
  cluster_input_target %=
    +(  qi::lit(to_name)
     >  common.whitespace
     >  common.port_addr
     >  common.line_end
     );

  cluster_input_block.name("cluster-input-spec");
  cluster_input_block %=
     (  common.opt_whitespace
     >> description_decl
     >> common.line_end
     >> common.opt_whitespace
     >> qi::lit(input_block_name)
     >  common.whitespace
     >  qi::lit(from_name)
     >  common.whitespace
     >  common.port_name
     >  common.line_end
     >  common.opt_whitespace
     >  cluster_input_target
     );

  cluster_output_block.name("cluster-output-spec");
  cluster_output_block %=
     (  common.opt_whitespace
     >> description_decl
     >> common.line_end
     >> common.opt_whitespace
     >> qi::lit(output_block_name)
     >  common.whitespace
     >  qi::lit(from_name)
     >  common.whitespace
     >  common.port_addr
     >  common.line_end
     >  common.opt_whitespace
     >  qi::lit(to_name)
     >  common.whitespace
     >  common.port_name
     >  common.line_end
     );

  cluster_block.name("cluster-block-spec");
  cluster_block %=
     (  common.opt_whitespace
     >> qi::lit(cluster_block_name)
     >  common.whitespace
     >  common.type_name
     >  common.line_end
     >  common.opt_whitespace
     >  description_decl
     >  common.line_end
     > *(  cluster_config_block
        |  cluster_input_block
        |  cluster_output_block
        )
     );

  cluster_block_set.name("cluster-blocks-set");
  cluster_block_set %=
     (  cluster_block
     > *(  common.config_block
        |  common.process_block
        |  common.connect_block
        )
     );
}

template <typename Iterator>
cluster_grammar<Iterator>
::~cluster_grammar()
{
}

// ==================================================================
class printer
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

void
print_info(std::ostream& ostr, boost::spirit::info const& what)
{
  printer pr(ostr);
  basic_info_walker<printer> walker(pr, what.tag, 0);
  boost::apply_visitor(walker, what.value);
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
  std::string const indent = std::string(depth * indent_width, ' ');

  m_ostr << indent;

  m_ostr << "tag: " << tag;
  if (!value.empty())
  {
    m_ostr << ", value: " << value;
  }
}

}
