/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_GRAVL_READ_VIDEO_GRAVL_PROCESS_H
#define VISTK_PROCESSES_GRAVL_READ_VIDEO_GRAVL_PROCESS_H

#include "gravl-config.h"

#include <vistk/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file read_video_gravl_process.h
 *
 * \brief Declaration of the read video gravl process.
 */

namespace vistk
{

/**
 * \class read_video_gravl_process
 *
 * \brief A process for reading in images from gravl RAF.
 *
 * \process Read images given a URI to a gravl resource.
 *
 * \oports
 *
 * \oport{image} The image read in for the step.
 *
 * \configs
 *
 * \config{pixtype} The pixel type of the input images.
 * \config{pixfmt} The pixel format of the input images.
 * \config{input} The URI of the resource.
 *
 * \reqs
 *
 * \req The \port{image} port must be connected.
 * \req The \key{input} configuration must be a valid URI.
 *
 * \ingroup process_image
 */
class VISTK_PROCESSES_GRAVL_NO_EXPORT read_video_gravl_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config The configuration for the process.
     */
    read_video_gravl_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~read_video_gravl_process();
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
     * \brief Step the process.
     */
    void _step();
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // VISTK_PROCESSES_GRAVL_READ_VIDEO_GRAVL_PROCESS_H
