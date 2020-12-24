// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PROCESSES_SHIFT_DETECTED_OBJECT_SET_FRAMES_PROCESS_H
#define SPROKIT_PROCESSES_SHIFT_DETECTED_OBJECT_SET_FRAMES_PROCESS_H

#include <sprokit/pipeline/process.h>

#include "kwiver_processes_export.h"

/**
 * \file shift_process.h
 *
 * \brief Declaration of the detected object set frame shift process
 */

namespace kwiver
{

/**
 * \class shift_detected_object_set_frames_process
 *
 * \brief Shifts input stream of detected object sets
 *
 * \process Either
 *
 * \iports
 *
 * \iports{any} The input detected object set
 *
 * \oports
 *
 * \oport{any} The ith - offset detected object set from the input stream.
 *             Any items outside the input stream bounds are provided as empty
 *             detected object sets
 * *
 * \reqs
 *
 * \req The \port{any} input must be connected.
 * \req The \port{any} output must be connected.
 *
 * \ingroup examples
 */
class KWIVER_PROCESSES_NO_EXPORT shift_detected_object_set_frames_process
  : public sprokit::process
{
  public:
    PLUGIN_INFO( "shift_detected_object_set",
                 "Shift an input stream of detected objects "
		 "by a certain number of frames")
    /**
     * \brief Constructor.
     *
     * \param config The configuration for the process.
     */
    shift_detected_object_set_frames_process
      (kwiver::vital::config_block_sptr const& config);
    /**
     * \brief Destructor.
     */
    ~shift_detected_object_set_frames_process();
  protected:
    /**
     * \brief Configure the process.
     */
    void _configure();

    /**
     * \brief Step the process.
     */
    void _step();
  private:
    void make_ports();
    void make_config();

    class priv;
    std::unique_ptr<priv> d;
};

}

#endif // SPROKIT_PROCESSES_SHIFT_DETECTED_OBJECT_SET_FRAMES_PROCESS_H
