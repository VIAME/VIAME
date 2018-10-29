/*ckwg +29
 * Copyright 2011-2017 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SPROKIT_PIPELINE_EDGE_H
#define SPROKIT_PIPELINE_EDGE_H

#include <sprokit/pipeline/sprokit_pipeline_export.h>

#include <vital/config/config_block.h>
#include <vital/noncopyable.h>
#include <vital/optional.h>

#include "types.h"

#ifdef WIN32
#pragma warning (push)
#pragma warning (disable : 4244)
#pragma warning (disable : 4267)
#endif
#include <boost/chrono/system_clocks.hpp>
#include <boost/operators.hpp>
#ifdef WIN32
#pragma warning (pop)
#endif

#include <vector>

/**
 * \file edge.h
 *
 * \brief Header for \link sprokit::edge edges\endlink.
 */

namespace sprokit {

/**
 * \class edge_datum_t <sprokit/pipeline/edge.h>
 *
 * \brief The packet of data that actually exists within edges.
 */
class SPROKIT_PIPELINE_EXPORT edge_datum_t
  : private boost::equality_comparable< edge_datum_t >
{
public:
  /**
   * \brief Constructor.
   */
  edge_datum_t();

  /**
   * \brief Constructor.
   *
   * \param datum_ The datum on the edge.
   * \param stamp_ The stamp for the datum.
   */
  edge_datum_t( datum_t const& datum_, stamp_t const& stamp_ );

  /**
   * \brief Destructor.
   */
  ~edge_datum_t();

  /**
   * \brief Compare two \ref edge_datum_t packets.
   *
   * \param rhs The second packet.
   *
   * \returns True if \p a and \p b are the same, false otherwise.
   */
  bool operator==( edge_datum_t const& rhs ) const;

  /// The datum on the edge.
  datum_t datum;
  /// The stamp for the datum.
  stamp_t stamp;
};

/// A typedef for a multiple packets which go through an \ref edge.
typedef std::vector< edge_datum_t > edge_data_t;
/// A group of \link edge edges\endlink.
typedef std::vector< edge_t > edges_t;


// ------------------------------------------------------------------
/**
 * \class edge edge.h <sprokit/pipeline/edge.h>
 *
 * \brief A connection between two \ref process ports which can carry data.
 *
 * \ingroup base_classes
 */
class SPROKIT_PIPELINE_EXPORT edge
  : private kwiver::vital::noncopyable
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
  edge( kwiver::vital::config_block_sptr const& config = kwiver::vital::config_block::empty_config() );

  /**
   * \brief Destructor.
   */
  ~edge();

  /**
   * \brief Query whether the edge represents a dependency from upstream to downstream.
   *
   * \returns True if the edge expresses that upstream must be executed before downstream, false otherwise.
   */
  bool makes_dependency() const;

  /**
   * \brief Query whether the edge has any data in it or not.
   *
   * \returns True if there is data available within the edge, false otherwise.
   */
  bool has_data() const;

  /**
   * \brief Query whether the edge can accept more data or not.
   *
   * \returns True if the edge can hold no more data, false otherwise.
   */
  bool full_of_data() const;

  /**
   * \brief Query how many results are in the edge.
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
  void push_datum( edge_datum_t const& datum );

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
   * \param idx The element in the queue to look at.
   *
   * \returns The next datum available from the edge.
   */
  edge_datum_t peek_datum( size_t idx = 0 ) const;

  /**
   * \brief Remove a datum from the edge.
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

  typedef boost::chrono::high_resolution_clock clock_t;
  typedef clock_t::duration duration_t;

  /**
   * \brief Push a datum into the edge.
   *
   * \see push_datum
   *
   * \param datum The datum to put into the edge.
   * \param duration The maximum amount of time to wait.
   */
  bool try_push_datum( edge_datum_t const& datum, duration_t const& duration );

  /**
   * \brief Extract a datum from the edge or fail if a timeout is reached.
   *
   * \see get_datum
   *
   * \param duration The maximum amount of time to wait.
   *
   * \returns The next datum available from the edge, or \c boost::none if the timeout was reached.
   */
  kwiver::vital::optional< edge_datum_t > try_get_datum( duration_t const& duration );

  /**
   * \brief Trigger the edge to flush all data and not accept any more data.
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
   * \brief Trigger the edge to flush all data and not accept any more data.
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
  void set_upstream_process( process_t process );

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
  void set_downstream_process( process_t process );

  /// Configuration that indicates the edge implies an execution dependency between upstream and downstream.
  static kwiver::vital::config_block_key_t const config_dependency;

  /// Configuration for the maximum capacity of an edge.
  static kwiver::vital::config_block_key_t const config_capacity;

  /// Configuration for edge blocking behaviour
  static kwiver::vital::config_block_key_t const config_blocking;


private:
  class SPROKIT_PIPELINE_NO_EXPORT priv;
  std::unique_ptr< priv > d;
};

}

template struct boost::equality_comparable< sprokit::edge_datum_t >;
#endif // SPROKIT_PIPELINE_EDGE_H
