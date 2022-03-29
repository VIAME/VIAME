/**
 * \file
 * \brief Append a detected object set to an object track set
 */

#ifndef VIAME_APPEND_DETECTION_TRACKS_PROCESS_H
#define VIAME_APPEND_DETECTION_TRACKS_PROCESS_H

#include <sprokit/pipeline/process.h>

#include <plugins/core/viame_processes_core_export.h>

#include <memory>

namespace viame
{

namespace core
{

// -----------------------------------------------------------------------------
/**
 * @brief Append a detected object set to an object track set
 */
class VIAME_PROCESSES_CORE_NO_EXPORT append_detections_to_tracks_process
  : public sprokit::process
{
public:
  // -- CONSTRUCTORS --
  append_detections_to_tracks_process( kwiver::vital::config_block_sptr const& config );
  virtual ~append_detections_to_tracks_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;

}; // end class append_detections_to_tracks_process

} // end namespace core
} // end namespace viame

#endif // VIAME_APPEND_DETECTION_TRACKS_PROCESS_H
