/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_UTILITIES_TIMESTAMP_READER_PROCESS_H
#define VISTK_PROCESSES_UTILITIES_TIMESTAMP_READER_PROCESS_H

#include "utilities-config.h"

#include <vistk/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file timestamp_reader_process.h
 *
 * \brief Declaration of the timestamp reader process.
 */

namespace vistk
{

/**
 * \class timestamp_reader_process
 *
 * \brief A process for reading timestamps from a file.
 *
 * \process Read timestamps from a file.
 *
 * \oports
 *
 * \oport{timestamp} The timestamp read from the file.
 *
 * \configs
 *
 * \config{path} The file to read.
 *
 * \reqs
 *
 * \req The \port{timestamp} output must be connected.
 */
class VISTK_PROCESSES_UTILITIES_NO_EXPORT timestamp_reader_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the process.
     */
    timestamp_reader_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~timestamp_reader_process();
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

#endif // VISTK_PROCESSES_UTILITIES_TIMESTAMP_READER_PROCESS_H
