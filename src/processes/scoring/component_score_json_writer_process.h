/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_SCORING_COMPONENT_SCORE_JSON_WRITER_PROCESS_H
#define VISTK_PROCESSES_SCORING_COMPONENT_SCORE_JSON_WRITER_PROCESS_H

#include "scoring-config.h"

#include <vistk/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file component_score_json_writer_process.h
 *
 * \brief A process which writes out component scores to a file in JSON.
 */

namespace vistk
{

/**
 * \class component_score_json_writer_process
 *
 * \brief A process which writes out component scores to a file in JSON.
 *
 * \process Writes component scores from a scoring process in JSON format.
 *
 * \iports
 *
 * \iport{score/\portvar{component}} The score for the \portvar{component} score.
 * \iport{stats/\portvar{component}} The statistics for the \portvar{component} score.
 *
 * \configs
 *
 * \oport{path} The path to output the scores into.
 *
 * \reqs
 *
 * \req At least one \port{score/\portvar{component}} port must be connected.
 *
 * \ingroup process_scoring
 */
class VISTK_PROCESSES_SCORING_NO_EXPORT component_score_json_writer_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the process.
     */
    component_score_json_writer_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~component_score_json_writer_process();
  protected:
    /**
     * \brief Configure the subclass.
     */
    void _configure();

    /**
     * \brief Initialize the process.
     */
    void _init();

    /**
     * \brief Step the subclass.
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

#endif // VISTK_PROCESSES_SCORING_COMPONENT_SCORE_JSON_WRITER_PROCESS_H
