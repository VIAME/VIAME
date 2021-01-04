// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef _KWIVER_MATCHER_PROCESS_H_
#define _KWIVER_MATCHER_PROCESS_H_

#include <sprokit/pipeline/process.h>
#include "kwiver_processes_export.h"

#include <memory>

namespace kwiver
{

// ----------------------------------------------------------------
/**
 * \class matcher_process
 *
 * \brief Stabilizes a series of image.
 *
 * \iports
 * \iport{timestamp}
 * \iport{image}
 * \iport{feature_set}
 * \iport{descriptor_set}
 *
 * \oports
 * \oport{feature_track_set}
 *
 */
class KWIVER_PROCESSES_NO_EXPORT matcher_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "feature_matcher",
               "Match extracted descriptors and detected features." )

  typedef sprokit::process base_t;

  matcher_process( kwiver::vital::config_block_sptr const& config );
    virtual ~matcher_process();

protected:
    virtual void _configure();
    virtual void _step();

private:
    void make_ports();
    void make_config();

    class priv;
    const std::unique_ptr<priv> d;
 }; // end class matcher_process

} // end namespace
#endif /* _KWIVER_MATCHER_PROCESS_H_ */
