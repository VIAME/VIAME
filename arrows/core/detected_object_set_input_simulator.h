// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface for detected_object_set_input_simulator
 */

#ifndef KWIVER_ARROWS_CORE_DETECTED_OBJECT_SET_INPUT_SIMULATOR_H
#define KWIVER_ARROWS_CORE_DETECTED_OBJECT_SET_INPUT_SIMULATOR_H

#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/algo/detected_object_set_input.h>

namespace kwiver {
namespace arrows {
namespace core {

class KWIVER_ALGO_CORE_EXPORT detected_object_set_input_simulator
  : public vital::algo::detected_object_set_input
{
public:
  // NOTE: Keep description in sync with detected_object_set_output_simulator
  PLUGIN_INFO( "simulator",
               "Detected object set reader using SIMULATOR format.\n\n"
               "Detection are generated algorithmicly." )

  detected_object_set_input_simulator();
  virtual ~detected_object_set_input_simulator();

  virtual vital::config_block_sptr get_configuration() const;
  virtual void set_configuration(vital::config_block_sptr config);
  virtual bool check_configuration(vital::config_block_sptr config) const;

  virtual void open( std::string const& filename );
  virtual bool read_set( kwiver::vital::detected_object_set_sptr & set, std::string& image_name );

private:
  class priv;
  std::unique_ptr< priv > d;
};

} } } // end namespace

#endif // KWIVER_ARROWS_CORE_DETECTED_OBJECT_SET_INPUT_SIMULATOR_H
