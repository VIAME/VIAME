/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_IMAGE_LAYERED_IMAGE_READER_PROCESS_H
#define VISTK_PROCESSES_IMAGE_LAYERED_IMAGE_READER_PROCESS_H

#include "image-config.h"

#include <vistk/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file layered_image_reader_process.h
 *
 * \brief A process which reads layered image files.
 */

namespace vistk
{

/**
 * \class layered_image_reader_process
 *
 * \brief A process which reads layered image files.
 *
 * \process A process which reads layered image files.
 *
 * \oports
 *
 * \oport{image/\portvar{layer}} The image layer named \portvar{tag}.
 *
 * \configs
 *
 * \config{pixtype} The pixel type of the images.
 * \config{pixfmt} The pixel format of the images.
 * \config{path} The file to read image format strings from.
 *
 * \reqs
 *
 * \req The output ports \port{image/\portvar{layer}} must be connected.
 *
 * \ingroup process_image
 */
class VISTK_PROCESSES_IMAGE_NO_EXPORT layered_image_reader_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the process.
     */
    layered_image_reader_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~layered_image_reader_process();
  protected:
    /**
     * \brief Configure the process.
     */
    void _configure();

    /**
     * \brief Collate data from the input edges.
     */
    void _step();

    /**
     * \brief Subclass output port information.
     *
     * \param port The port to get information about.
     *
     * \returns Information about an output port.
     */
    port_info_t _output_port_info(port_t const& port);
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // VISTK_PROCESSES_IMAGE_LAYERED_IMAGE_READER_PROCESS_H
