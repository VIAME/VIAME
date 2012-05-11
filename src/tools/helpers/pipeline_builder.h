/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline_util/pipe_bakery.h>

#include <vistk/pipeline/types.h>

#include <vistk/utilities/path.h>

#include <istream>
#include <string>

class pipeline_builder
{
  public:
    pipeline_builder();
    ~pipeline_builder();

    void load_pipeline(std::istream& istr);
    void load_supplement(vistk::path_t const& path);
    void add_setting(std::string const& setting);

    vistk::pipeline_t pipeline() const;
    vistk::config_t config() const;
  private:
    vistk::pipe_blocks m_blocks;
};
