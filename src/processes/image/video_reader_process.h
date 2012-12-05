/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_IMAGE_VIDEO_READER_PROCESS_H
#define VISTK_PROCESSES_IMAGE_VIDEO_READER_PROCESS_H

#include "image-config.h"

#include <vistk/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file video_reader_process.h
 *
 * \brief Declaration of the video reader process.
 */

namespace vistk
{

/**
 * \class video_reader_process
 *
 * \brief A process for reading frames from a file.
 *
 * \process Read image frames from a video.
 *
 * \oports
 *
 * \oport{image} The frame.
 * \oport{timestamp} The timestamp for the image.
 *
 * \configs
 *
 * \config{pixtype} The pixel type of the input images.
 * \config{pixfmt} The pixel format of the input images.
 * \config{input} The video file to read.
 * \config{verify} Verify frames during initialization.
 * \config{impl} The implementation of readers to use.
 *
 * \reqs
 *
 * \req The \port{image} output must be connected.
 * \req The \key{input} configuration must be a valid filepath.
 *
 * \ingroup process_image
 */
class VISTK_PROCESSES_IMAGE_NO_EXPORT video_reader_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config The configuration for the process.
     */
    video_reader_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~video_reader_process();
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

#endif // VISTK_PROCESSES_IMAGE_VIDEO_READER_PROCESS_H
