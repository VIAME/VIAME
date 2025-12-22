/**
 * \file
 * \brief Split an object track set into a feature_track_set and a landmark_map
 */

#ifndef VIAME_CORE_SPLIT_OBJECT_TRACK_TO_FEATURE_LANDMARK_PROCESS_H
#define VIAME_CORE_SPLIT_OBJECT_TRACK_TO_FEATURE_LANDMARK_PROCESS_H

#include <sprokit/pipeline/process.h>

#include <plugins/core/viame_processes_core_export.h>

#include <memory>

namespace viame
{

namespace core
{

// -----------------------------------------------------------------------------
/**
 * @brief Split an object track set into a feature_track_set and a landmark_map
 */
class VIAME_PROCESSES_CORE_NO_EXPORT split_object_track_to_feature_landmark_process
  : public sprokit::process
{
public:
  // -- CONSTRUCTORS --
  split_object_track_to_feature_landmark_process( kwiver::vital::config_block_sptr const& config );
  virtual ~split_object_track_to_feature_landmark_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;

}; // end class split_object_track_to_feature_landmark_process

} // end namespace core
} // end namespace viame

#endif // VIAME_CORE_SPLIT_OBJECT_TRACK_TO_FEATURE_LANDMARK_PROCESS_H
