// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VITAL_UTIL_DIRECTED_GRAPH_H
#define VITAL_UTIL_DIRECTED_GRAPH_H

#include <algorithm>
#include <deque>
#include <stdexcept>
#include <vector>

namespace kwiver {

namespace vital {

/**
 * \class not_a_dag_exception directed_graph.h <vital/util/directed_graph.h>
 *
 * \brief Exception thrown when a graph contains cycles and is not a DAG.
 */
class not_a_dag_exception
  : public std::runtime_error
{
public:
  not_a_dag_exception()
    : std::runtime_error( "Graph contains a cycle and is not a DAG" )
  {}
};

/**
 * \class directed_graph directed_graph.h <vital/util/directed_graph.h>
 *
 * \brief A simple directed graph with vertex properties.
 *
 * This provides a minimal implementation of a directed graph to replace
 * boost::adjacency_list for the specific use case of topological sorting.
 * Vertices are identified by size_t indices and can store arbitrary data.
 *
 * \tparam VertexProperty The type of data stored at each vertex.
 */
template < typename VertexProperty >
class directed_graph
{
public:
  /// Type used to identify vertices
  typedef std::size_t vertex_descriptor;

  /// Type for storing edges (adjacency list per vertex)
  typedef std::vector< vertex_descriptor > adjacency_list_t;

  /// Structure holding vertex data
  struct vertex_data
  {
    VertexProperty property;
    adjacency_list_t out_edges;
  };

  /// Default constructor - creates an empty graph
  directed_graph() = default;

  /**
   * \brief Add a new vertex to the graph.
   *
   * \return The descriptor (index) of the newly added vertex.
   */
  vertex_descriptor
  add_vertex()
  {
    vertex_descriptor v = m_vertices.size();
    m_vertices.emplace_back();
    return v;
  }

  /**
   * \brief Add an edge from source to target vertex.
   *
   * \param source The source vertex descriptor.
   * \param target The target vertex descriptor.
   */
  void
  add_edge( vertex_descriptor source, vertex_descriptor target )
  {
    m_vertices[ source ].out_edges.push_back( target );
  }

  /**
   * \brief Access the property of a vertex.
   *
   * \param v The vertex descriptor.
   * \return Reference to the vertex property.
   */
  VertexProperty&
  operator[]( vertex_descriptor v )
  {
    return m_vertices[ v ].property;
  }

  /**
   * \brief Access the property of a vertex (const version).
   *
   * \param v The vertex descriptor.
   * \return Const reference to the vertex property.
   */
  VertexProperty const&
  operator[]( vertex_descriptor v ) const
  {
    return m_vertices[ v ].property;
  }

  /**
   * \brief Get the number of vertices in the graph.
   *
   * \return The number of vertices.
   */
  std::size_t
  num_vertices() const
  {
    return m_vertices.size();
  }

  /**
   * \brief Get the outgoing edges from a vertex.
   *
   * \param v The vertex descriptor.
   * \return Const reference to the list of target vertices.
   */
  adjacency_list_t const&
  out_edges( vertex_descriptor v ) const
  {
    return m_vertices[ v ].out_edges;
  }

private:
  std::vector< vertex_data > m_vertices;
};

/**
 * \brief Perform a topological sort on a directed graph.
 *
 * This function performs a depth-first search based topological sort.
 * The result is output via an output iterator in reverse topological order
 * (i.e., vertices with no outgoing edges come first in the output).
 *
 * \tparam VertexProperty The vertex property type of the graph.
 * \tparam OutputIterator The type of output iterator.
 *
 * \param graph The directed graph to sort.
 * \param out The output iterator to receive sorted vertices.
 *
 * \throws not_a_dag_exception if the graph contains a cycle.
 */
template < typename VertexProperty, typename OutputIterator >
void
topological_sort(
  directed_graph< VertexProperty > const& graph,
  OutputIterator out )
{
  std::size_t const n = graph.num_vertices();

  // Color markers for DFS: 0 = white (unvisited), 1 = gray (in progress), 2 =
  // black (done)
  std::vector< int > color( n, 0 );

  // Result container (we build it in reverse order, then output)
  std::deque< typename directed_graph< VertexProperty >::vertex_descriptor >
  result;

  // DFS visit function (iterative to avoid stack overflow on large graphs)
  // We use a stack-based approach that simulates recursion
  struct stack_entry
  {
    typename directed_graph< VertexProperty >::vertex_descriptor vertex;
    std::size_t edge_index;
  };

  std::vector< stack_entry > stack;

  for( std::size_t start = 0; start < n; ++start )
  {
    if( color[ start ] != 0 )
    {
      continue; // Already visited
    }

    stack.push_back( { start, 0 } );
    color[ start ] = 1; // Mark as in-progress

    while( !stack.empty() )
    {
      auto& current = stack.back();
      auto const& edges = graph.out_edges( current.vertex );

      // Find next unvisited neighbor
      bool found_unvisited = false;
      while( current.edge_index < edges.size() )
      {
        auto neighbor = edges[ current.edge_index ];
        ++current.edge_index;

        if( color[ neighbor ] == 1 )
        {
          // Back edge found - cycle detected
          throw not_a_dag_exception();
        }
        else if( color[ neighbor ] == 0 )
        {
          // Unvisited - push to stack
          color[ neighbor ] = 1;
          stack.push_back( { neighbor, 0 } );
          found_unvisited = true;
          break;
        }
        // color[neighbor] == 2: already finished, skip
      }

      if( !found_unvisited )
      {
        // All neighbors processed, finish this vertex
        color[ current.vertex ] = 2;
        result.push_back( current.vertex );
        stack.pop_back();
      }
    }
  }

  // Output results
  for( auto const& v : result )
  {
    *out++ = v;
  }
}

} // namespace vital

} // namespace kwiver

#endif // VITAL_UTIL_DIRECTED_GRAPH_H
