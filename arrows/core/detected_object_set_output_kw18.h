// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface for detected_object_set_output_kw18
 */

#ifndef KWIVER_ARROWS_DETECTED_OBJECT_SET_OUTPUT_KW18_H
#define KWIVER_ARROWS_DETECTED_OBJECT_SET_OUTPUT_KW18_H

#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/algo/detected_object_set_output.h>

#include <memory>

namespace kwiver {
namespace arrows {
namespace core {

class KWIVER_ALGO_CORE_EXPORT detected_object_set_output_kw18
  : public vital::algo::detected_object_set_output
{
public:
  // NOTE: Keep description in sync with detected_object_set_input_kw18
  PLUGIN_INFO( "kw18",
    "Detected object set writer using kw18 format.\n\n"
    "  - Column(s) 1: Track-id\n"
    "  - Column(s) 2: Track-length (number of detections)\n"
    "  - Column(s) 3: Frame-number (-1 if not available)\n"
    "  - Column(s) 4-5: Tracking-plane-loc(x,y) (could be same as World-loc)\n"
    "  - Column(s) 6-7: Velocity(x,y)\n"
    "  - Column(s) 8-9: Image-loc(x,y)\n"
    "  - Column(s) 10-13: Img-bbox(TL_x,TL_y,BR_x,BR_y)"
    " (location of top-left & bottom-right vertices)\n"
    "  - Column(s) 14: Area\n"
    "  - Column(s) 15-17: World-loc(x,y,z)"
    " (longitude, latitude, 0 - when available)\n"
    "  - Column(s) 18: Timesetamp (-1 if not available)\n"
    "  - Column(s) 19: Track-confidence (-1 if not available)" )

  detected_object_set_output_kw18();
  virtual ~detected_object_set_output_kw18();

  virtual vital::config_block_sptr get_configuration() const;
  virtual void set_configuration( vital::config_block_sptr config );
  virtual bool check_configuration( vital::config_block_sptr config ) const;

  virtual void write_set( const kwiver::vital::detected_object_set_sptr set, std::string const& image_name );

private:
  class priv;
  std::unique_ptr< priv > d;
};

} } } // end namespace

#endif // KWIVER_ARROWS_DETECTED_OBJECT_SET_OUTPUT_KW18_H
