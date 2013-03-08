/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_FILELIST_FILELIST_WRITER_PROCESS_H
#define VISTK_PROCESSES_FILELIST_FILELIST_WRITER_PROCESS_H

#include "filelist-config.h"

#include <vistk/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file filelist_writer_process.h
 *
 * \brief Declaration of the filelist writer process.
 */

namespace vistk
{

/**
 * \class filelist_writer_process
 *
 * \brief A process for writing paths to files.
 *
 * \process Write paths to files.
 *
 * \iports
 *
 * \iport{path} The path to write.
 *
 * \configs
 *
 * \config{output} The file to write filepaths to.
 *
 * \reqs
 *
 * \req The \port{path} port must be connected.
 *
 * \ingroup process_filelist
 */
class VISTK_PROCESSES_FILELIST_NO_EXPORT filelist_writer_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config The configuration for the process.
     */
    filelist_writer_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~filelist_writer_process();
  protected:
    /**
     * \brief Configure the process.
     */
    void _configure();

    /**
     * \brief Reset the process.
     */
    void _reset();

    /**
     * \brief Step the process.
     */
    void _step();
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // VISTK_PROCESSES_FILELIST_FILELIST_WRITER_PROCESS_H
