/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_IMAGE_IMAGE_READER_PROCESS_H
#define VISTK_PROCESSES_IMAGE_IMAGE_READER_PROCESS_H

#include "image-config.h"

#include <vistk/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file image_reader_process.h
 *
 * \brief Declaration of the image reader process.
 */

namespace vistk
{

/**
 * \class image_reader_process
 *
 * \brief A process which reads in a list of images from a file.
 *
 * \process A process for reading images from files.
 *
 * \oports
 *
 * \oport{image} The image read in for the step.
 *
 * \configs
 *
 * \config{pixtype} The pixel type of the input images.
 * \config{pixfmt} The pixel format of the input images.
 * \config{input} The file to read filepaths from.
 * \config{verify} Verify images during initialization.
 *
 * \reqs
 *
 * \req The \port{image} port must be connected.
 * \req The \key{input} configuration must be a valid filepath.
 */
class VISTK_PROCESSES_IMAGE_NO_EXPORT image_reader_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the process.
     */
    image_reader_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~image_reader_process();
  protected:
    /**
     * \brief Configure the process.
     */
    void _configure();

    /**
     * \brief Initialize the process.
     */
    void _init();

    /**
     * \brief Reads the next image.
     */
    void _step();
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // VISTK_PROCESSES_IMAGE_IMAGE_READER_PROCESS_H
