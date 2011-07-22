/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "load_pipe.h"

#include <vistk/pipeline/pipeline.h>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#include <fstream>
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

  fin.open(fname.c_str());

  if (fin.fail())
  {
    /// \todo Throw an exception.
  }

  pipe_blocks blocks = load_pipe_blocks(fin, fname.parent_path());

  fin.close();

  return blocks;
}

pipe_blocks
load_pipe_blocks(std::istream& istr, boost::filesystem::path const& inc_root)
{
  pipe_blocks blocks;
  std::stringstream sstr;

  flatten_pipe_declaration(sstr, istr, inc_root);

  /// \todo What parser do we want to use here?

  return blocks;
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
bake_pipe_blocks(pipe_blocks const& blocks)
{
  /// \todo Bake pipe blocks into a pipeline.
}

void
flatten_pipe_declaration(std::stringstream& sstr, std::istream& istr, boost::filesystem::path const& inc_root)
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
      boost::filesystem::path file_path(line.substr(include_directive.size()));

      if (file_path.is_relative())
      {
        file_path = inc_root / file_path;

        /// \todo Support system include directories?
      }

      boost::system::error_code ec;

      if (!boost::filesystem::exists(file_path, ec))
      {
        /// \todo Throw an exception.
      }

      /// \todo Check ec.

      std::ifstream fin;

      fin.open(file_path.native().c_str());

      if (fin.fail())
      {
        /// \todo Throw an exception.
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

  if (istr.fail())
  {
    /// \todo Throw an exception.
  }
}

}
