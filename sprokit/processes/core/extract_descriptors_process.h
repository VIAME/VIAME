// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef _KWIVER_EXTRACT_DESCRIPTORS_PROCESS_H_
#define _KWIVER_EXTRACT_DESCRIPTORS_PROCESS_H_

#include <sprokit/pipeline/process.h>
#include "kwiver_processes_export.h"

#include <memory>

namespace kwiver
{

// ----------------------------------------------------------------
/**
 * \class extract_descriptors_process
 *
 * \brief Stabilizes a series of image.
 *
 * \iports
 * \iport{timestamp}
 * \iport{image}
 *
 * \oports
 * \oport{src_to_ref_homography}
 *
 */
class KWIVER_PROCESSES_NO_EXPORT extract_descriptors_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "extract_descriptors",
               "Extract descriptors from detected features." )

  typedef sprokit::process base_t;

  extract_descriptors_process( kwiver::vital::config_block_sptr const& config );
    virtual ~extract_descriptors_process();

protected:
    virtual void _configure();
    virtual void _step();

private:
    void make_ports();
    void make_config();

    class priv;
    const std::unique_ptr<priv> d;
 }; // end class extract_descriptors_process

} // end namespace
#endif /* _KWIVER_EXTRACT_DESCRIPTORS_PROCESS_H_ */
