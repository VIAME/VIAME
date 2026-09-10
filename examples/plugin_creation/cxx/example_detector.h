/*
 * INSERT COPYRIGHT STATEMENT OR DELETE THIS
 */

#ifndef VIAME_EXAMPLE_DETECTOR_H
#define VIAME_EXAMPLE_DETECTOR_H

#include <viame/algorithm_framework/algo/image_object_detector.h>

namespace viame {

// An algorithm is a subclass of the interface it implements. PLUGGABLE_IMPL
// declares the constructor, the configuration accessors and one member per
// parameter: `text` below becomes `c_text`, with the default and the
// description the pipeline's `--help` prints.
class example_detector
  : public kwiver::vital::algo::image_object_detector
{
public:
  PLUGGABLE_IMPL(
    example_detector,
    "Example externally created plugin.",

    PARAM_DEFAULT(
      text, std::string,
      "Text to display to user.",
      "External Plugin C++ Example" )
  );

  virtual ~example_detector() = default;

  // Called once the configuration is in place; the pipeline rejects the
  // process if this returns false.
  bool check_configuration(
    kwiver::vital::config_block_sptr config ) const override;

  // Main detection method
  kwiver::vital::detected_object_set_sptr detect(
    kwiver::vital::image_container_sptr image_data ) const override;
};

} // end namespace

#endif /* VIAME_EXAMPLE_DETECTOR_H */
