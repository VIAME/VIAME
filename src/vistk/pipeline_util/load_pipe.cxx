/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "load_pipe.h"

#include <vistk/pipeline/pipeline.h>

/**
 * \file load_pipe.cxx
 *
 * \brief Implementation of the pipeline declaration loading.
 */

namespace vistk
{

pipeline_t
load_pipe_file(boost::filesystem::path const& fname)
{
  /// \todo Load the pipeline from a file.
}

pipeline_t
load_pipe(std::istream& istr)
{
  pipeline_t pipe;

  /// \todo What parser do we want to use here?

  return pipe;
}

}
