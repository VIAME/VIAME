/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_EDGES_EXAMPLES_DUMB_PIPE_EDGE_H
#define VISTK_EDGES_EXAMPLES_DUMB_PIPE_EDGE_H

#include "examples-config.h"

#include <vistk/pipeline/edge.h>

namespace vistk
{

/**
 * \class dumb_pipe_edge
 *
 * \brief A connection between two process ports which can carry data.
 */
class VISTK_EDGES_EXAMPLES_NO_EXPORT dumb_pipe_edge
  : public edge
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the edge.
     */
    dumb_pipe_edge(config_t const& config);
    /**
     * \brief Destructor.
     */
    virtual ~dumb_pipe_edge();

    /**
     * \brief Whether the edge can accept more data or not.
     */
    virtual bool full_of_data() const;
    /**
     * \brief How many results are in the edge.
     */
    virtual size_t datum_count() const;

    /**
     * \brief Push a datum into the edge.
     *
     * \note Some edge implementations may block if they are full.
     *
     * \param datum The datum to put into the edge.
     */
    virtual void push_datum(edge_datum_t const& datum);
    /**
     * \brief Look at the next datum in the edge.
     *
     * \note Some edge implementations may block if they are empty.
     */
    virtual edge_datum_t peek_datum();
  protected:
    /**
     * \brief Removes a datum from the edge.
     */
    virtual void pop_datum();
  private:
    class priv;
    const boost::shared_ptr<priv> d;
};

} // end namespace vistk

#endif // VISTK_EDGES_EXAMPLES_DUMB_PIPE_EDGE_H
