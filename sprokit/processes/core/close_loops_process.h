// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef _KWIVER_CLOSE_LOOPS_PROCESS_H_
#define _KWIVER_CLOSE_LOOPS_PROCESS_H_

#include <sprokit/pipeline/process.h>
#include "kwiver_processes_export.h"

#include <memory>

namespace kwiver
{

// ----------------------------------------------------------------
/**
 * \class close_loops_process
 *
 * \brief closes loops in a set of images
 *
 * \iports
 * \iport{timestamp}
 * \iport{next_tracks}
 * \iport{loop_back_tracks}
 *
 * \oports
 * \oport{feature_track_set}
 *
 */
class KWIVER_PROCESSES_NO_EXPORT close_loops_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "close_loops",
               "Detects loops in a track set using features with descriptors.")

  typedef sprokit::process base_t;

  close_loops_process( kwiver::vital::config_block_sptr const& config );
    virtual ~close_loops_process();

protected:
    virtual void _configure();
    virtual void _step();

private:
    void make_ports();
    void make_config();

    class priv;
    const std::unique_ptr<priv> d;
 }; // end class detect_features_process

} // end namespace
#endif /* _KWIVER_CLOSE_LOOPS_PROCESS_H_ */
