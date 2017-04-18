/*ckwg +29
 * Copyright 2011-2013 by Kitware, Inc.
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

#include "load_pipe.h"
#include "load_pipe_exception.h"

#if defined(_WIN32) || defined(_WIN64)
#include <sprokit/pipeline_util/include-paths.h>
#endif
#include "path.h"
#include "pipe_grammar.h"

#include <vital/vital_foreach.h>

#include <sprokit/pipeline/pipeline.h>
#include <sprokit/pipeline/utils.h>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/filesystem/operations.hpp>

#include <istream>
#include <sstream>
#include <string>

/**
 * \file load_pipe.cxx
 *
 * \brief Implementation of the pipeline declaration loading.
 */

namespace sprokit
{

static std::string const default_include_dirs = std::string(DEFAULT_PIPE_INCLUDE_PATHS);
static envvar_name_t const sprokit_include_envvar = envvar_name_t("SPROKIT_PIPE_INCLUDE_PATH");
static std::string const include_directive = "!include ";
static char const comment_marker = '#';

static void flatten_pipe_declaration(std::stringstream& sstr, std::istream& istr, path_t const& inc_root);
static bool is_separator(char ch);


// ------------------------------------------------------------------
pipe_blocks
load_pipe_blocks_from_file(path_t const& fname)
{
  std::stringstream sstr;
  std::string const str = fname.string<std::string>();

  sstr << include_directive << str;

  pipe_blocks const blocks = load_pipe_blocks(sstr, fname.parent_path());

  return blocks;
}


// ------------------------------------------------------------------
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


// ------------------------------------------------------------------
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


// ------------------------------------------------------------------
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


// ------------------------------------------------------------------
void
flatten_pipe_declaration(std::stringstream& sstr, std::istream& istr, path_t const& inc_root)
{
  typedef path_t include_path_t;
  typedef std::vector<include_path_t> include_paths_t;

  include_paths_t include_dirs;

  // Build include directories.
  {
    include_dirs.push_back(inc_root);

    include_paths_t include_dirs_tmp;

    envvar_value_t const extra_include_dirs = get_envvar(sprokit_include_envvar);

    if (extra_include_dirs)
    {
      boost::split(include_dirs_tmp, *extra_include_dirs, is_separator, boost::token_compress_on);

      include_dirs.insert(include_dirs.end(), include_dirs_tmp.begin(), include_dirs_tmp.end());
    }

    boost::split(include_dirs_tmp, default_include_dirs, is_separator, boost::token_compress_on);

    include_dirs.insert(include_dirs.end(), include_dirs_tmp.begin(), include_dirs_tmp.end());
  }

  std::string line;

  while (std::getline(istr, line))
  {
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
        VITAL_FOREACH (include_path_t const& include_dir, include_dirs)
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


// ------------------------------------------------------------------
bool
is_separator(char ch)
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
