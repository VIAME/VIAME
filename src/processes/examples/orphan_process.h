/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_EXAMPLES_ORPHAN_PROCESS_H
#define VISTK_PROCESSES_EXAMPLES_ORPHAN_PROCESS_H

#include "examples-config.h"

#include <vistk/pipeline/process.h>

/**
 * \file orphan_process.h
 *
 * \brief Declaration of the orphan process.
 */

namespace vistk
{

/**
 * \class orphan_process
 *
 * \brief A process which does approximately nothing.
 *
 * \process An orphan process.
 */
class VISTK_PROCESSES_EXAMPLES_NO_EXPORT orphan_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the process.
     */
    orphan_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~orphan_process();
};

}

#endif // VISTK_PROCESSES_EXAMPLES_ORPHAN_PROCESS_H
