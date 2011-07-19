/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_EDGE_H
#define VISTK_PIPELINE_EDGE_H

#include "pipeline-config.h"

#include "types.h"

#include <boost/tuple/tuple.hpp>
#include <boost/utility.hpp>

#include <string>
#include <map>
#include <vector>

/**
 * \file edge.h
 *
 * \brief Header for \link edge edges\endlink.
 */

namespace vistk
{

/// A typedef for a single packet which goes through an \ref edge.
typedef boost::tuple<datum_t, stamp_t> edge_datum_t;
/// A group of \link edge edges\endlink.
typedef std::vector<edge_t> edges_t;

/**
 * \class edge edge.h <vistk/pipeline/edge.h>
 *
 * \brief A connection between two process ports which can carry data.
 *
 * \ingroup base_classes
 */
class VISTK_PIPELINE_EXPORT edge
  : boost::noncopyable
{
  public:
    /**
     * \brief Destructor.
     */
    virtual ~edge();

    /**
     * \brief Whether the edge represents a dependency from upstream to downstream.
     *
     * \returns True if the edge expresses that upstream must be executed before downstream, false otherwise.
     */
    virtual bool makes_dependency() const;

    /**
     * \brief Whether the edge has any data in it or not.
     *
     * \returns True if there is data available within the edge, false otherwise.
     */
    virtual bool has_data() const;
    /**
     * \brief Whether the edge can accept more data or not.
     *
     * \returns True if the edge can hold no more data, false otherwise.
     */
    virtual bool full_of_data() const = 0;
    /**
     * \brief How many results are in the edge.
     *
     * \returns The number of data items the edge holds.
     */
    virtual size_t datum_count() const = 0;

    /**
     * \brief Push a datum into the edge.
     *
     * \note Some edge implementations may block if they are full.
     *
     * \param datum The datum to put into the edge.
     */
    virtual void push_datum(edge_datum_t const& datum) = 0;
    /**
     * \brief Extract a datum from the edge.
     *
     * This calls \ref edge::peek_datum and then removes it from the edge.
     *
     * \note Some edge implementations may block if they are empty.
     *
     * \returns The next datum available from the edge.
     */
    virtual edge_datum_t get_datum();
    /**
     * \brief Look at the next datum in the edge.
     *
     * \note Some edge implementations may block if they are empty.
     *
     * \returns The next datum available from the edge.
     */
    virtual edge_datum_t peek_datum() = 0;

    /**
     * \brief Set whether the data the edge delivers is required for downstream.
     *
     * \param required Whether the data the edge delivers is required for downstream.
     */
    void set_required_by_downstream(bool required);
    /**
     * \brief Whether the data the edge delivers is required for downstream.
     *
     * \returns True if the edge carries required data for downstream, false otherwise.
     */
    bool required_by_downstream() const;

    /**
     * \brief Set the process which is connected to the input side of the edge.
     *
     * \throws null_process_connection Thrown if \p process is \c NULL.
     * \throws input_already_connected Thrown if a process is already connected.
     *
     * \param process The process which can push data into the edge.
     */
    virtual void set_upstream_process(process_t process);
    /**
     * \brief Set the process which is connected to the output side of the edge.
     *
     * \throws null_process_connection Thrown if \p process is \c NULL.
     * \throws output_already_connected Thrown if a process is already connected.
     *
     * \param process The process which can pull data from the edge.
     */
    virtual void set_downstream_process(process_t process);
  protected:
    /**
     * \brief Constructor.
     *
     * \param config Contains configuration for the edge.
     */
    edge(config_t const& config);

    /**
     * \brief Removes a datum from the edge.
     */
    virtual void pop_datum() = 0;

    /// Whether the edge is required for the downstream process to run.
    bool m_required;

    /// The process that can put data into the edge.
    process_t m_upstream;
    /// The process that can pull data from the edge.
    process_t m_downstream;
};

} // end namespace vistk

#endif // VISTK_PIPELINE_EDGE_H
