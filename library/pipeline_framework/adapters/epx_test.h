// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_EMBEDDED_PIPELINE_EPX_TEST_H
#define SPROKIT_EMBEDDED_PIPELINE_EPX_TEST_H

#include "viame/pipeline_framework/adapters/viame_epx_test_export.h"

#include "embedded_pipeline_extension.h"

namespace viame {

// ----------------------------------------------------------------
/**
 * @brief
 *
 */
class VIAME_EPX_TEST_NO_EXPORT epx_test
  : public embedded_pipeline_extension
{
public:
  PLUGIN_INFO( "test",
               "Embedded Pipeline Extension used for testing" );

  // -- CONSTRUCTORS --
  epx_test();
  virtual ~epx_test() = default;

  void pre_setup( context& ctxt ) override;
  void end_of_output( context& ctxt ) override;
  void configure( viame::config_block_sptr const conf ) override;
  viame::config_block_sptr get_configuration() const override;

}; // end class epx_test

} // namespace viame

#endif // SPROKIT_EMBEDDED_PIPELINE_EPX_TEST_H
