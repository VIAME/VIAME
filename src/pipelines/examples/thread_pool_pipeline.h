/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINES_EXAMPLES_THREAD_POOL_PIPELINE_H
#define VISTK_PIPELINES_EXAMPLES_THREAD_POOL_PIPELINE_H

#include "examples-config.h"

#include <vistk/pipeline/pipeline.h>

#include <boost/shared_ptr.hpp>

namespace vistk
{

/**
 * \class thread_pool_pipeline
 *
 * \brief A pipeline which process execution among a group of threads.
 *
 * \section config Configuration
 *
 * \li \c num_threads The number of threads to run. A setting of \c 0 means "auto".
 */
class VISTK_PIPELINES_EXAMPLES_NO_EXPORT thread_pool_pipeline
  : public pipeline
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the edge.
     * \param num_threads The number of threads to use. 0 means the number of processors available.
     */
    thread_pool_pipeline(config_t const& config, size_t num_threads);
    /**
     * \brief Destructor.
     */
    virtual ~thread_pool_pipeline();

    /**
     * \brief Runs the pipeline.
     */
    virtual void run();

    /**
     * \brief Shuts the pipeline down.
     */
    virtual void shutdown();
  protected:
    /// The number of threads to run.
    size_t const m_num_threads;
  private:
    class priv;
    boost::shared_ptr<priv> d;
};

} // end namespace vistk

#endif // VISTK_PIPELINES_EXAMPLES_THREAD_POOL_PIPELINE_H
