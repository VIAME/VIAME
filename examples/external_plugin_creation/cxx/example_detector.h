/*
 * INSERT COPYRIGHT STATEMENT OR DELETE THIS
 */

#ifndef VIAME_EXAMPLE_DETECTOR_H
#define VIAME_EXAMPLE_DETECTOR_H

#include <vital/algo/image_object_detector.h>

namespace viame {

class example_detector :
  public kwiver::vital::algorithm_impl<
    example_detector, kwiver::vital::algo::image_object_detector >
{
public:
  example_detector();
  virtual ~example_detector();

  // Get the current configuration (parameters) for this detector
  virtual kwiver::vital::config_block_sptr get_configuration() const;

  // Set configurations automatically parsed from input pipeline and config files
  virtual void set_configuration( kwiver::vital::config_block_sptr config );
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  // Main detection method
  virtual kwiver::vital::detected_object_set_sptr detect(
    kwiver::vital::image_container_sptr image_data ) const;

private:
  class priv;
  const std::unique_ptr< priv > d;
};

} // end namespace

#endif /* VIAME_EXAMPLE_DETECTOR_H */
