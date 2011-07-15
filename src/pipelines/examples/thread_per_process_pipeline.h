/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINES_EXAMPLES_THREAD_PER_PROCESS_PIPELINE_H
#define VISTK_PIPELINES_EXAMPLES_THREAD_PER_PROCESS_PIPELINE_H

#include "examples-config.h"

#include <vistk/pipeline/pipeline.h>

#include <boost/shared_ptr.hpp>

namespace vistk
{

/**
 * \class thread_per_process_pipeline
 *
 * \brief A pipeline which runs each process in its own thread.
 */
class VISTK_PIPELINES_EXAMPLES_NO_EXPORT thread_per_process_pipeline
  : public pipeline
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the edge.
     */
    thread_per_process_pipeline(config_t const& config);
    /**
     * \brief Destructor.
     */
    virtual ~thread_per_process_pipeline();

    /**
     * \brief Runs the pipeline.
     */
    virtual void run();

    /**
     * \brief Shuts the pipeline down.
     */
    virtual void shutdown();
  private:
    class priv;
    boost::shared_ptr<priv> d;
};

} // end namespace vistk

#endif // VISTK_PIPELINES_EXAMPLES_THREAD_PER_PROCESS_PIPELINE_H
