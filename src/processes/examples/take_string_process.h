/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_EXAMPLES_TAKE_STRING_PROCESS_H
#define VISTK_PROCESSES_EXAMPLES_TAKE_STRING_PROCESS_H

#include "examples-config.h"

#include <vistk/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file take_string_process.h
 *
 * \brief Declaration of the string taking process.
 */

namespace vistk
{

/**
 * \class take_string_process
 *
 * \brief A process which takes strings.
 *
 * \process Takes strings.
 *
 * \iports
 *
 * \iport{string} The source of strings to take.
 *
 * \reqs
 *
 * \req The \port{string} port must be connected.
 *
 * \ingroup examples
 */
class VISTK_PROCESSES_EXAMPLES_NO_EXPORT take_string_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config The configuration for the process.
     */
    take_string_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~take_string_process();
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

#endif // VISTK_PROCESSES_EXAMPLES_TAKE_STRING_PROCESS_H
