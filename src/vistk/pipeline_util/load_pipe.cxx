/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "load_pipe.h"
#include "load_pipe_exception.h"

#if defined(_WIN32) || defined(_WIN64)
#include <vistk/pipeline_util/include-paths.h>
#endif
#include "pipe_grammar.h"

#include <vistk/pipeline/pipeline.h>
#include <vistk/pipeline/utils.h>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/foreach.hpp>

#include <istream>
#include <sstream>
#include <string>

/**
 * \file load_pipe.cxx
 *
 * \brief Implementation of the pipeline declaration loading.
 */

namespace vistk
{

namespace
{

typedef path_t include_path_t;
typedef std::vector<include_path_t> include_paths_t;

}

static std::string const default_include_dirs = std::string(DEFAULT_PIPE_INCLUDE_PATHS);
static envvar_name_t const vistk_include_envvar = envvar_name_t("VISTK_PIPE_INCLUDE_PATH");
static std::string const include_directive = "!include ";
static char const comment_marker = '#';

static void flatten_pipe_declaration(std::stringstream& sstr, std::istream& istr, path_t const& inc_root);
static bool is_separator(char ch);

pipe_blocks
load_pipe_blocks_from_file(path_t const& fname)
{
  std::stringstream sstr;
  path_t::string_type const& fstr = fname.native();
  std::string const str(fstr.begin(), fstr.end());

  sstr << include_directive << str;

  pipe_blocks blocks = load_pipe_blocks(sstr, fname.parent_path());

  return blocks;
}

pipe_blocks
load_pipe_blocks(std::istream& istr, path_t const& inc_root)
{
  std::stringstream sstr;

  sstr.exceptions(std::stringstream::failbit | std::stringstream::badbit);

  try
  {
    flatten_pipe_declaration(sstr, istr, inc_root);
  }
  catch (std::ios_base::failure const& e)
  {
    throw stream_failure_exception(e.what());
  }

  return parse_pipe_blocks_from_string(sstr.str());
}

cluster_blocks
load_cluster_blocks_from_file(path_t const& fname)
{
  std::stringstream sstr;
  path_t::string_type const& fstr = fname.native();
  std::string const str(fstr.begin(), fstr.end());

  sstr << include_directive << str;

  cluster_blocks const blocks = load_cluster_blocks(sstr, fname.parent_path());

  return blocks;
}

cluster_blocks
load_cluster_blocks(std::istream& istr, path_t const& inc_root)
{
  std::stringstream sstr;

  sstr.exceptions(std::stringstream::failbit | std::stringstream::badbit);

  try
  {
    flatten_pipe_declaration(sstr, istr, inc_root);
  }
  catch (std::ios_base::failure const& e)
  {
    throw stream_failure_exception(e.what());
  }

  return parse_cluster_blocks_from_string(sstr.str());
}

void
flatten_pipe_declaration(std::stringstream& sstr, std::istream& istr, path_t const& inc_root)
{
  include_paths_t include_dirs;

  // Build include directories.
  {
    include_dirs.push_back(inc_root);

    include_paths_t include_dirs_tmp;

    envvar_value_t const extra_include_dirs = get_envvar(vistk_include_envvar);

    if (extra_include_dirs)
    {
      boost::split(include_dirs_tmp, *extra_include_dirs, is_separator, boost::token_compress_on);

      include_dirs.insert(include_dirs.end(), include_dirs_tmp.begin(), include_dirs_tmp.end());
    }

    boost::split(include_dirs_tmp, default_include_dirs, is_separator, boost::token_compress_on);

    include_dirs.insert(include_dirs.end(), include_dirs_tmp.begin(), include_dirs_tmp.end());
  }

  while (istr.good())
  {
    std::string line;

    std::getline(istr, line);

    boost::trim_left(line);

    if (line.empty())
    {
      continue;
    }

    if (boost::starts_with(line, include_directive))
    {
      path_t file_path(line.substr(include_directive.size()));

      boost::system::error_code ec;

      if (file_path.is_relative())
      {
        BOOST_FOREACH (include_path_t const& include_dir, include_dirs)
        {
          path_t const inc_file_path = include_dir / file_path;

          if (boost::filesystem::exists(inc_file_path, ec))
          {
            file_path = inc_file_path;
            break;
          }
        }
      }

      if (!boost::filesystem::exists(file_path, ec))
      {
        throw file_no_exist_exception(file_path);
      }

      /// \todo Check ec.

      if (!boost::filesystem::is_regular_file(file_path, ec))
      {
        throw not_a_file_exception(file_path);
      }

      /// \todo Check ec.

      boost::filesystem::ifstream fin;

      fin.open(file_path);

      if (!fin.good())
      {
        throw file_open_exception(file_path);
      }

      flatten_pipe_declaration(sstr, fin, file_path.parent_path());

      fin.close();
    }
    /// \todo: Support comments not starting in column 1?
    else if (line[0] == comment_marker)
    {
      continue;
    }
    else
    {
      sstr << line << std::endl;
    }
  }
}

bool is_separator(char ch)
{
  char const separator =
#if defined(_WIN32) || defined(_WIN64)
    ';';
#else
    ':';
#endif

  return (ch == separator);
}

}
