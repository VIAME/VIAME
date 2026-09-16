/*
 * INSERT COPYRIGHT STATEMENT OR DELETE THIS
 */

#include "example_detector.h"

#include <viame/core_types/detected_object_set.h>

#include <iostream>

namespace viame {

// ----------------------------------------------------------------------------
bool
external_example_detector
::check_configuration( viame::config_block_sptr /*config*/ ) const
{
  return !c_text.empty();
}

// ----------------------------------------------------------------------------
viame::detected_object_set_sptr
external_example_detector
::detect( viame::image_container_sptr /*image_data*/ ) const
{
  auto detected_set = std::make_shared< viame::detected_object_set >();

  std::cout << "Text: " << c_text << std::endl;

  return detected_set;
}

} // end namespace
