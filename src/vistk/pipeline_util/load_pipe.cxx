/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "load_pipe.h"

#include <vistk/pipeline/pipeline.h>

#include <fstream>

/**
 * \file load_pipe.cxx
 *
 * \brief Implementation of the pipeline declaration loading.
 */

namespace vistk
{

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

}
