/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_EDGE_H
#define VISTK_PIPELINE_EDGE_H

#include "pipeline-config.h"

#include "config.h"
#include "types.h"

#include <boost/tuple/tuple.hpp>
#include <boost/noncopyable.hpp>
#include <boost/operators.hpp>
#include <boost/scoped_ptr.hpp>

#include <vector>

/**
 * \file edge.h
 *
 * \brief Header for \link vistk::edge edges\endlink.
 */

namespace vistk
{

/// A typedef for a single packet which goes through an \ref edge.
typedef boost::tuple<datum_t, stamp_t> edge_datum_t;
/// A typedef for a multiple packets which go through an \ref edge.
typedef std::vector<edge_datum_t> edge_data_t;
/// A group of \link edge edges\endlink.
typedef std::vector<edge_t> edges_t;

/**
 * \class edge edge.h <vistk/pipeline/edge.h>
 *
 * \brief A connection between two \ref process ports which can carry data.
 *
 * \ingroup base_classes
 */
class VISTK_PIPELINE_EXPORT edge
  : boost::noncopyable
{
  public:
    /**
     * \brief Constructor.
     *
     * \preconds
     *
     * \precond{\p config}
     *
     * \endpreconds
     *
     * \param config Contains configuration for the edge.
     */
    edge(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~edge();

    /**
     * \brief Whether the edge represents a dependency from upstream to downstream.
     *
     * \returns True if the edge expresses that upstream must be executed before downstream, false otherwise.
     */
    bool makes_dependency() const;

    /**
     * \brief Whether the edge has any data in it or not.
     *
     * \returns True if there is data available within the edge, false otherwise.
     */
    bool has_data() const;
    /**
     * \brief Whether the edge can accept more data or not.
     *
     * \returns True if the edge can hold no more data, false otherwise.
     */
    bool full_of_data() const;
    /**
     * \brief How many results are in the edge.
     *
     * \returns The number of data items the edge holds.
     */
    size_t datum_count() const;

    /**
     * \brief Push a datum into the edge.
     *
     * \note This call blocks if \c full_of_data is \c true.
     *
     * \postconds
     *
     * \postcond{<code>this->has_data() == true</code>}
     * \postcond{The edge has one more datum packet in it.}
     * \postcond{\c this takes ownership of \p datum.}
     *
     * \endpostconds
     *
     * \param datum The datum to put into the edge.
     */
    void push_datum(edge_datum_t const& datum);
    /**
     * \brief Extract a datum from the edge.
     *
     * \note This call blocks if \c has_data is \c false.
     *
     * \throws datum_requested_after_complete Thrown if called after \ref mark_downstream_as_complete.
     *
     * \preconds
     *
     * \precond{<code>this->is_downstream_complete() == false</code>}
     *
     * \endpreconds
     *
     * \postconds
     *
     * \postcond{<code>this->full_of_data() == false</code>}}
     * \postcond{The edge has one less datum packet in it.}
     * \postcond{The caller takes ownership of the returned datum packet.}
     *
     * \endpostconds
     *
     * \returns The next datum available from the edge.
     */
    edge_datum_t get_datum();
    /**
     * \brief Look at the next datum in the edge.
     *
     * \note This call blocks if \c has_data is \c false.
     *
     * \preconds
     *
     * \precond{<code>this->is_downstream_complete() == false</code>}
     *
     * \endpreconds
     *
     * \throws datum_requested_after_complete Thrown if called after \ref mark_downstream_as_complete.
     *
     * \postconds
     *
     * \postcond{The edge has the same number of data packets as before.}
     * \postcond{The edge retains ownership of the datum packet.}
     *
     * \endpostconds
     *
     * \returns The next datum available from the edge.
     */
    edge_datum_t peek_datum();
    /**
     * \brief Removes a datum from the edge.
     *
     * \preconds
     *
     * \precond{<code>this->is_downstream_complete() == false</code>}
     *
     * \endpreconds
     *
     * \throws datum_requested_after_complete Thrown if called after \ref mark_downstream_as_complete.
     *
     * \postconds
     *
     * \postcond{<code>this->full_of_data() == false</code>}
     * \postcond{The edge has one less datum packet in it.}
     *
     * \endpostconds
     */
    void pop_datum();

    /**
     * \brief Triggers the edge to flush all data and not accept any more data.
     *
     * \postconds
     *
     * \postcond{<code>this->is_downstream_complete() == true</code>}
     * \postcond{<code>this->has_data() == false</code>}
     *
     * \endpostconds
     */
    void mark_downstream_as_complete();
    /**
     * \brief Triggers the edge to flush all data and not accept any more data.
     *
     * \returns True if the downstream process indicated that no more data is required, false otherwise.
     */
    bool is_downstream_complete() const;

    /**
     * \brief Set the process which is connected to the input side of the edge.
     *
     * \preconds
     *
     * \precond{\p process}
     * \precond{An upstream process is not already set.}
     *
     * \endpreconds
     *
     * \throws null_process_connection_exception Thrown if \p process is \c NULL.
     * \throws input_already_connected_exception Thrown if a process is already connected.
     *
     * \param process The process which can push data into the edge.
     */
    void set_upstream_process(process_t process);
    /**
     * \brief Set the process which is connected to the output side of the edge.
     *
     * \preconds
     *
     * \precond{\p process}
     * \precond{A downstream process is not already set.}
     *
     * \endpreconds
     *
     * \throws null_process_connection_exception Thrown if \p process is \c NULL.
     * \throws output_already_connected_exception Thrown if a process is already connected.
     *
     * \param process The process which can pull data from the edge.
     */
    void set_downstream_process(process_t process);

    /// Configuration key that indicates the edge implies an execution dependency between upstream and downstream.
    static config::key_t const config_dependency;
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

/**
 * \brief Compare two \ref edge_datum_t packets.
 *
 * \param a The first packet.
 * \param b The second packet.
 *
 * \returns True if \p a and \p b are the same, false otherwise.
 */
bool VISTK_PIPELINE_EXPORT operator == (edge_datum_t const& a, edge_datum_t const& b);

}

template struct boost::equality_comparable<vistk::edge_datum_t>;

#endif // VISTK_PIPELINE_EDGE_H
