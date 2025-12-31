/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Custom ITK registration process
 */

#ifndef VIAME_ITK_EO_IR_REGISTRATION_PROCESS_H
#define VIAME_ITK_EO_IR_REGISTRATION_PROCESS_H

#include <sprokit/pipeline/process.h>

#include "viame_itk_export.h"
#include "../core/align_multimodal_imagery_process.h"

#include <memory>

namespace viame
{

namespace itk
{

// -----------------------------------------------------------------------------
/**
 * @brief Register optical and thermal imagery using ITK
 */
class VIAME_ITK_NO_EXPORT itk_eo_ir_registration_process
  : public viame::core::align_multimodal_imagery_process
{
public:
  // -- CONSTRUCTORS --
  itk_eo_ir_registration_process( kwiver::vital::config_block_sptr const& config );
  virtual ~itk_eo_ir_registration_process();

protected:
  virtual void attempt_registration( const buffered_frame& frame1,
                                     const buffered_frame& frame2,
                                     const bool output_frame1_time );

}; // end class itk_eo_ir_registration_process

} // end namespace itk
} // end namespace viame

#endif // VIAME_ITK_EO_IR_REGISTRATION_PROCESS_H
