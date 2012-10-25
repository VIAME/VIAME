/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_FILELIST_FILELIST_READER_PROCESS_H
#define VISTK_PROCESSES_FILELIST_FILELIST_READER_PROCESS_H

#include "filelist-config.h"

#include <vistk/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file filelist_reader_process.h
 *
 * \brief Declaration of the filelist reader process.
 */

namespace vistk
{

/**
 * \class filelist_reader_process
 *
 * \brief A process for reading in a list of paths from a file.
 *
 * \process Read paths from a file.
 *
 * \oports
 *
 * \oport{path} The path from the file.
 *
 * \configs
 *
 * \config{input} The file to read filepaths from.
 *
 * \reqs
 *
 * \req The \port{path} port must be connected.
 * \req The \key{input} configuration must be a valid filepath.
 *
 * \ingroup process_image
 */
class VISTK_PROCESSES_FILELIST_NO_EXPORT filelist_reader_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config The configuration for the process.
     */
    filelist_reader_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~filelist_reader_process();
  protected:
    /**
     * \brief Configure the process.
     */
    void _configure();

    /**
     * \brief Step the process.
     */
    void _step();
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // VISTK_PROCESSES_FILELIST_FILELIST_READER_PROCESS_H
