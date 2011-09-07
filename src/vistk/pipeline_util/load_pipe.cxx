/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "load_pipe.h"
#include "load_pipe_exception.h"

#include "pipe_grammar.h"

#include <vistk/pipeline/pipeline.h>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/filesystem/operations.hpp>

#include <fstream>
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

static std::string const include_directive = "!include ";
static char const comment_marker = '#';

static void flatten_pipe_declaration(std::stringstream& sstr, std::istream& istr, path_t const& inc_root);

pipe_blocks
load_pipe_blocks_from_file(path_t const& fname)
{
  std::stringstream sstr;
  path_t::string_type const fstr = fname.native();
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
  catch (std::ios_base::failure& e)
  {
    throw stream_failure_exception(e.what());
  }

  return parse_pipe_blocks_from_string(sstr.str());
}

void
flatten_pipe_declaration(std::stringstream& sstr, std::istream& istr, path_t const& inc_root)
{
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

      if (file_path.is_relative())
      {
        file_path = inc_root / file_path;

        /// \todo Support system include directories?
      }

      boost::system::error_code ec;

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

      std::ifstream fin;

      fin.open(file_path.native().c_str());

      if (!fin.good())
      {
        throw file_open_exception(file_path);
      }

      flatten_pipe_declaration(sstr, fin, inc_root);

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

}
