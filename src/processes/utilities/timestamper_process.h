/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_UTILITIES_TIMESTAMPER_PROCESS_H
#define VISTK_PROCESSES_UTILITIES_TIMESTAMPER_PROCESS_H

#include "utilities-config.h"

#include <vistk/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file timestamper_process.h
 *
 * \brief Declaration of the timestamper process.
 */

namespace vistk
{

/**
 * \class timestamper_process
 *
 * \brief A process for generating timestamps.
 *
 * \process A process for generating timestamps.
 *
 * \iports
 *
 * \iport{color} The color to use for the output stamps.
 *
 * \oports
 *
 * \oport{timestamp} The timestamp read from the file.
 *
 * \configs
 *
 * \config{start_frame} The frame number to start at.
 * \config{start_time} The time to start .
 * \config{frame_rate} The number of frames per second.
 *
 * \reqs
 *
 * \req The \port{timestamp} output must be connected.
 */
class VISTK_PROCESSES_UTILITIES_NO_EXPORT timestamper_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the process.
     */
    timestamper_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~timestamper_process();
  protected:
    /**
     * \brief Checks the output port connections and the configuration.
     */
    void _init();

    /**
     * \brief Pushes a new timestamp through the output port.
     */
    void _step();
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // VISTK_PROCESSES_UTILITIES_TIMESTAMPER_PROCESS_H
