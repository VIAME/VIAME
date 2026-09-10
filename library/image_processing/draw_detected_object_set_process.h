// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/pipeline_framework/process.h>
#include "viame_processes_image_processing_export.h"

#include <memory>

namespace kwiver {

// ----------------------------------------------------------------
/**
 * \class  draw_detected_object_set
 *
 * \brief Instantiate and run draw_detected_object_set algorithm
 *
 * \iports
 * \iport{image}
 * \iport{detected_object_set}
 *
 * \oports
 * \oport{image}
 *
 */
class VIAME_PROCESSES_IMAGE_PROCESSING_EXPORT draw_detected_object_set_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "draw_detected_object_set",
               "Draws border around detected objects in the set using the selected algorithm.\n\n"
               "This process is a wrapper around a `draw_detected_object_set` algorithm.")

  draw_detected_object_set_process( kwiver::vital::config_block_sptr const& config );
  virtual ~draw_detected_object_set_process();

protected:
  void _configure(); // preconnection
  void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  std::unique_ptr< priv > d;
};   // end class

} // end namespace
