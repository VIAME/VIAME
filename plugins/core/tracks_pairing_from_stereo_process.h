/**
 * \file
 * \brief Compute object tracks pair from stereo depth map information
 */

#ifndef VIAME_TRACKS_PAIRING_FROM_STEREO_PROCESS_H
#define VIAME_TRACKS_PAIRING_FROM_STEREO_PROCESS_H

#include <sprokit/pipeline/process.h>

#include <plugins/core/viame_processes_core_export.h>

#include <memory>

namespace viame
{

namespace core
{

// -----------------------------------------------------------------------------
/**
 * @brief Compute object tracks pair from stereo depth map information
 */
class VIAME_PROCESSES_CORE_NO_EXPORT tracks_pairing_from_stereo_process
  : public sprokit::process
{
public:
  // -- CONSTRUCTORS --
  tracks_pairing_from_stereo_process( kwiver::vital::config_block_sptr const& config );
  virtual ~tracks_pairing_from_stereo_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;

}; // end class tracks_pairing_from_stereo_process

} // end namespace core
} // end namespace viame

#endif // VIAME_TRACKS_PAIRING_FROM_STEREO_PROCESS_H
