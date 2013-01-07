/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_UTILITIES_INVERT_TRANSFORM_PROCESS_H
#define VISTK_PROCESSES_UTILITIES_INVERT_TRANSFORM_PROCESS_H

#include "utilities-config.h"

#include <vistk/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file invert_transform_process.h
 *
 * \brief Declaration of the invert transform process.
 */

namespace vistk
{

/**
 * \class invert_transform_process
 *
 * \brief A process for inverting transforms.
 *
 * \process Invert tranformation matrices.
 *
 * \iports
 *
 * \iport{transform} The transform to invert.
 *
 * \oports
 *
 * \oport{inv_transform} The inverse of the input transform.
 *
 * \reqs
 *
 * \req The \port{transform} and \port{inv_transform} ports must be connected.
 *
 * \ingroup process_utilities
 */
class VISTK_PROCESSES_UTILITIES_NO_EXPORT invert_transform_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config The configuration for the process.
     */
    invert_transform_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~invert_transform_process();
  protected:
    /**
     * \brief Step the process.
     */
    void _step();
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // VISTK_PROCESSES_UTILITIES_INVERT_TRANSFORM_PROCESS_H
