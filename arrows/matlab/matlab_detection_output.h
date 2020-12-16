// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header defining matlab image object set writer
 */

#ifndef KWIVER_VITAL_BINDINGS_MATLAB_DETECTION_OUTPUT_H_
#define KWIVER_VITAL_BINDINGS_MATLAB_DETECTION_OUTPUT_H_

#include <vital/algo/detected_object_set_output.h>
#include <arrows/matlab/kwiver_algo_matlab_export.h>

namespace kwiver {
namespace arrows {
namespace matlab {

class KWIVER_ALGO_MATLAB_EXPORT matlab_detection_output
  : public vital::algo::detected_object_set_output
{
public:
  matlab_detection_output();
  virtual ~matlab_detection_output();

  PLUGIN_INFO( "matlab",
               "Bridge to matlab detection output writer.")

  virtual vital::config_block_sptr get_configuration() const;
  virtual void set_configuration(vital::config_block_sptr config);
  virtual bool check_configuration(vital::config_block_sptr config) const;

  virtual void write_set( const kwiver::vital::detected_object_set_sptr set, std::string const& image_name );

private:
  class priv;
  std::unique_ptr< priv > d;
};

} } } // end namespace

#endif
