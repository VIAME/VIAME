/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_IMAGE_IMAGE_READER_PROCESS_H
#define VISTK_PROCESSES_IMAGE_IMAGE_READER_PROCESS_H

#include "image-config.h"

#include <vistk/pipeline/process.h>

namespace vistk
{

/**
 * \class image_reader_process
 *
 * \brief A process which reads in a list of images from a file.
 *
 * \process A process for reading images from files.
 *
 * \iports
 *
 * \iport{color} The color to use for output stamps.
 *
 * \oports
 *
 * \oport{image} The image read in for the step.
 *
 * \configs
 *
 * \config{pixtype} The type of image to read.
 * \config{grayscale} Whether the inputs are grayscale or not.
 * \config{input} The file to read filepaths from.
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
     * \brief Initialize the process.
     */
    void _init();

    /**
     * \brief Reads the next image.
     */
    void _step();
  private:
    class priv;
    boost::shared_ptr<priv> d;
};

}

#endif // VISTK_PROCESSES_IMAGE_IMAGE_READER_PROCESS_H
