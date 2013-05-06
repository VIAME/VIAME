/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef SPROKIT_PROCESSES_EXAMPLES_ORPHAN_PROCESS_H
#define SPROKIT_PROCESSES_EXAMPLES_ORPHAN_PROCESS_H

#include "examples-config.h"

#include <sprokit/pipeline/process.h>

/**
 * \file orphan_process.h
 *
 * \brief Declaration of the orphan process.
 */

namespace sprokit
{

/**
 * \class orphan_process
 *
 * \brief A no-op process.
 *
 * \process A no-op process.
 *
 * \ingroup examples
 */
class SPROKIT_PROCESSES_EXAMPLES_NO_EXPORT orphan_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config The configuration for the process.
     */
    orphan_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~orphan_process();
};

}

#endif // SPROKIT_PROCESSES_EXAMPLES_ORPHAN_PROCESS_H
