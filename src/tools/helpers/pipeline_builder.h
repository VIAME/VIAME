/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_TOOLS_HELPERS_PIPELINE_BUILDER_H
#define VISTK_TOOLS_HELPERS_PIPELINE_BUILDER_H

#include <vistk/pipeline_util/path.h>
#include <vistk/pipeline_util/pipe_bakery.h>

#include <vistk/pipeline/types.h>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>

#include <istream>
#include <string>

class pipeline_builder
{
  public:
    pipeline_builder(boost::program_options::variables_map const& vm, boost::program_options::options_description const& desc);
    pipeline_builder();
    ~pipeline_builder();

    void load_pipeline(std::istream& istr);

    void load_from_options(boost::program_options::variables_map const& vm);

    void load_supplement(vistk::path_t const& path);
    void add_setting(std::string const& setting);

    vistk::pipeline_t pipeline() const;
    vistk::config_t config() const;
    vistk::pipe_blocks blocks() const;
  private:
    vistk::pipe_blocks m_blocks;
};

boost::program_options::options_description pipeline_common_options();
boost::program_options::options_description pipeline_input_options();
boost::program_options::options_description pipeline_output_options();
boost::program_options::options_description pipeline_run_options();

#endif // VISTK_TOOLS_HELPERS_PIPELINE_BUILDER_H
