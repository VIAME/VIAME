/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef SPROKIT_TOOLS_PIPELINE_BUILDER_H
#define SPROKIT_TOOLS_PIPELINE_BUILDER_H

#include "tools-config.h"

#include <sprokit/pipeline_util/path.h>
#include <sprokit/pipeline_util/pipe_bakery.h>

#include <sprokit/pipeline/types.h>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/noncopyable.hpp>

#include <istream>
#include <string>

namespace sprokit
{

class SPROKIT_TOOLS_EXPORT pipeline_builder
  : boost::noncopyable
{
  public:
    pipeline_builder(boost::program_options::variables_map const& vm, boost::program_options::options_description const& desc);
    pipeline_builder();
    ~pipeline_builder();

    void load_pipeline(std::istream& istr);

    void load_from_options(boost::program_options::variables_map const& vm);

    void load_supplement(sprokit::path_t const& path);
    void add_setting(std::string const& setting);

    sprokit::pipeline_t pipeline() const;
    sprokit::config_t config() const;
    sprokit::pipe_blocks blocks() const;
  private:
    sprokit::pipe_blocks m_blocks;
};

SPROKIT_TOOLS_EXPORT boost::program_options::options_description pipeline_common_options();
SPROKIT_TOOLS_EXPORT boost::program_options::options_description pipeline_input_options();
SPROKIT_TOOLS_EXPORT boost::program_options::options_description pipeline_output_options();
SPROKIT_TOOLS_EXPORT boost::program_options::options_description pipeline_run_options();

}

#endif // SPROKIT_TOOLS_PIPELINE_BUILDER_H
