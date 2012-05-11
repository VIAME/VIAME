/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_IMAGE_COMBINE_MASKS_PROCESS_H
#define VISTK_PROCESSES_IMAGE_COMBINE_MASKS_PROCESS_H

#include "image-config.h"

#include <vistk/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file combine_masks_process.h
 *
 * \brief A process which combines multiple masks into one.
 */

namespace vistk
{

/**
 * \class combine_masks_process
 *
 * \brief A process which combines multiple masks into one.
 *
 * \process Combine multiple masks into one mask.
 *
 * \iports
 *
 * \iport{mask/\portvar{tag}} The input mask named \portvar{tag}.
 *
 * \oports
 *
 * \oport{mask} The resulting mask file.
 *
 * \reqs
 *
 * \req The input ports \port{mask/\portvar{tag}} must be connected.
 * \req At least two input masks must be provided.
 *
 * \todo Add support for \c and and \c xor combinations.
 */
class VISTK_PROCESSES_IMAGE_NO_EXPORT combine_masks_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the process.
     */
    combine_masks_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~combine_masks_process();
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
     * \brief Collate data from the input edges.
     */
    void _step();

    /**
     * \brief Subclass input port information.
     *
     * \param port The port to get information about.
     *
     * \returns Information about an output port.
     */
    port_info_t _input_port_info(port_t const& port);
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // VISTK_PROCESSES_IMAGE_COMBINE_MASKS_PROCESS_H
