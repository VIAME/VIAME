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
#include <boost/filesystem/path.hpp>

#include <fstream>
#include <ios>
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

static void flatten_pipe_declaration(std::stringstream& sstr, std::istream& istr, boost::filesystem::path const& inc_root);

pipe_blocks
load_pipe_blocks_from_file(boost::filesystem::path const& fname)
{
  std::ifstream fin;

  boost::system::error_code ec;

  if (!boost::filesystem::exists(fname, ec))
  {
    throw file_no_exist_exception(fname);
  }

  /// \todo Check ec.

  fin.open(fname.c_str());

  if (fin.fail())
  {
    throw file_open_exception(fname);
  }

  pipe_blocks blocks = load_pipe_blocks(fin, fname.parent_path());

  fin.close();

  return blocks;
}

pipe_blocks
load_pipe_blocks(std::istream& istr, boost::filesystem::path const& inc_root)
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

pipeline_t
bake_pipe_from_file(boost::filesystem::path const& fname)
{
  return bake_pipe_blocks(load_pipe_blocks_from_file(fname));
}

pipeline_t
bake_pipe(std::istream& istr, boost::filesystem::path const& inc_root)
{
  return bake_pipe_blocks(load_pipe_blocks(istr, inc_root));
}

pipeline_t
bake_pipe_blocks(pipe_blocks const& /*blocks*/)
{
  pipeline_t pipe;

  /// \todo Bake pipe blocks into a pipeline.

  return pipe;
}

void
flatten_pipe_declaration(std::stringstream& sstr, std::istream& istr, boost::filesystem::path const& inc_root)
{
  istr.exceptions(std::istream::failbit | std::istream::badbit);

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
      boost::filesystem::path file_path(line.substr(include_directive.size()));

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

      std::ifstream fin;

      fin.open(file_path.native().c_str());

      if (fin.fail())
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
