// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef _KWIVER_DETECT_FEATURES_IF_KEYFRAME_PROCESS_H_
#define _KWIVER_DETECT_FEATURES_IF_KEYFRAME_PROCESS_H_

#include <sprokit/pipeline/process.h>
#include "kwiver_processes_export.h"

#include <memory>

namespace kwiver
{

class KWIVER_PROCESSES_NO_EXPORT detect_features_if_keyframe_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "detect_features_if_keyframe",
               "Detects feautres in an image if it is a keyframe.")

  typedef sprokit::process base_t;

  detect_features_if_keyframe_process(
    kwiver::vital::config_block_sptr const& config );

    virtual ~detect_features_if_keyframe_process();

protected:
    virtual void _configure();
    virtual void _step();

private:
    void make_ports();
    void make_config();

    class priv;
    const std::unique_ptr<priv> d;
 }; // end class detect_features_if_keyframe_process

} // end namespace
#endif /* _KWIVER_DETECT_FEATURES_IF_KEYFRAME_PROCESS_H_ */
