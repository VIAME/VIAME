/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_IMAGE_GRAYSCALE_PROCESS_H
#define VISTK_PROCESSES_IMAGE_GRAYSCALE_PROCESS_H

#include "image-config.h"

#include <vistk/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file grayscale_process.h
 *
 * \brief Declaration of the grayscale process.
 */

namespace vistk
{

/**
 * \class grayscale_process
 *
 * \brief A process which converts an image to grayscale.
 *
 * \process A process for converting an image into grayscale.
 *
 * \iports
 *
 * \iport{rgbimage} An RGB image to convert to grayscale.
 *
 * \oports
 *
 * \oport{grayimage} The input image in grayscale.
 *
 * \configs
 *
 * \config{pixtype} The type of image to convert.
 *
 * \reqs
 *
 * \req The \port{rgbimage} and \port{grayimage} ports must be connected.
 */
class VISTK_PROCESSES_IMAGE_NO_EXPORT grayscale_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the process.
     */
    grayscale_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~grayscale_process();
  protected:
    /**
     * \brief Initialize the process.
     */
    void _init();

    /**
     * \brief Turns an image into grayscale.
     */
    void _step();
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // VISTK_PROCESSES_IMAGE_GRAYSCALE_PROCESS_H
