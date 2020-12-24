// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "shift_detected_object_set_frames_process.h"

#include <vital/config/config_block.h>
#include <sprokit/processes/kwiver_type_traits.h>
#include <sprokit/pipeline/process_exception.h>

/**
 * \file shift_detected_object_set_frames_process.cxx
 *
 * \brief Implementation of the detected object set frame shift process
 */

namespace kwiver
{

create_config_trait( offset, int, "0", "The offset to shift the input by." );

class shift_detected_object_set_frames_process::priv
{
  public:
    priv(int offset);
    ~priv() = default;

    int remaining_offset;

    static vital::detected_object_set_sptr const empty_detected_object_set_sptr;
};

vital::detected_object_set_sptr const
shift_detected_object_set_frames_process::priv::empty_detected_object_set_sptr =
  std::make_shared<vital::detected_object_set>();

shift_detected_object_set_frames_process
::shift_detected_object_set_frames_process(vital::config_block_sptr const& config)
  : process(config)
  , d()
{
  make_ports();
  make_config();
}

shift_detected_object_set_frames_process
::~shift_detected_object_set_frames_process()
{
}

void
shift_detected_object_set_frames_process
::make_config()
{
  declare_config_using_trait( offset );
}

void
shift_detected_object_set_frames_process
::make_ports()
{
  sprokit::process::port_flags_t required;

  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( detected_object_set, required );

  // -- output --
  declare_output_port_using_trait( detected_object_set, required );
}

void
shift_detected_object_set_frames_process
::_configure()
{
  // Configure the process.
  {
    int offset = config_value_using_trait( offset );

    d.reset(new priv(offset));
  }
}

void
shift_detected_object_set_frames_process
::_step()
{
  while (d->remaining_offset > 0)
  {
    // Source data until the remaining offset is 0.
    push_to_port_using_trait( detected_object_set,
			      d->empty_detected_object_set_sptr);
    --d->remaining_offset;
  }

  if (d->remaining_offset < 0)
  {
    // Sink data util the remaining offset is 0.
    (void)grab_from_port_using_trait( detected_object_set );
    ++d->remaining_offset;
  }
  else
  {
    push_to_port_using_trait( detected_object_set,
			      grab_from_port_using_trait( detected_object_set ) );
  }
}

shift_detected_object_set_frames_process::priv
::priv(int offset)
  : remaining_offset(offset)
{
}

}
