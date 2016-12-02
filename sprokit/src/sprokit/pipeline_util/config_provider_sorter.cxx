/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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

/**
 * @file   config_provider_sorter.cxx
 * @brief  Implementation for config_provider_sorter class.
 */

#include "config_provider_sorter.h"
#include "pipe_bakery_exception.h"

#include <boost/graph/topological_sort.hpp>
#include <boost/foreach.hpp>


namespace sprokit {

// ------------------------------------------------------------------
config_provider_sorter
::config_provider_sorter()
  : m_vertex_map()
  , m_graph()
{
}


config_provider_sorter
::~config_provider_sorter()
{
}


// ------------------------------------------------------------------
kwiver::vital::config_block_keys_t
config_provider_sorter
::sorted() const
{
  vertices_t vertices;

  try
  {
    boost::topological_sort(m_graph, std::back_inserter(vertices));
  }
  catch (boost::not_a_dag const&)
  {
    throw circular_config_provide_exception();
  }

  kwiver::vital::config_block_keys_t keys;

  BOOST_FOREACH (vertex_t const& vertex, vertices)
  {
    node_t const& node = m_graph[vertex];

    if (node.deref)
    {
      keys.push_back(node.name);
    }
  }

  return keys;
}


// ------------------------------------------------------------------
void
config_provider_sorter
::operator () (kwiver::vital::config_block_key_t const& /*key*/, kwiver::vital::config_block_value_t const& /*value*/) const
{
}


// ------------------------------------------------------------------
void
config_provider_sorter
::operator () (kwiver::vital::config_block_key_t const& key, bakery_base::provider_request_t const& request)
{
  config_provider_t const& provider = request.first;
  kwiver::vital::config_block_value_t const& value = request.second;

  if (provider != bakery_base::provider_config)
  {
    return;
  }

  kwiver::vital::config_block_key_t const& target_key = kwiver::vital::config_block_key_t(value);

  typedef std::pair<vertex_map_t::iterator, bool> insertion_t;

  insertion_t from_iter = m_vertex_map.insert(vertex_entry_t(key, vertex_t()));
  insertion_t to_iter = m_vertex_map.insert(vertex_entry_t(target_key, vertex_t()));

  bool const& from_inserted = from_iter.second;
  bool const& to_inserted = to_iter.second;

  vertex_map_t::iterator& from = from_iter.first;
  vertex_map_t::iterator& to = to_iter.first;

  vertex_t& from_vertex = from->second;
  vertex_t& to_vertex = to->second;

  if (from_inserted)
  {
    from_vertex = boost::add_vertex(m_graph);
    m_graph[from_vertex].name = key;
  }

  if (to_inserted)
  {
    to_vertex = boost::add_vertex(m_graph);
    m_graph[to_vertex].name = target_key;
  }

  m_graph[from_vertex].deref = true;

  boost::add_edge(from_vertex, to_vertex, m_graph);
}

// ------------------------------------------------------------------
config_provider_sorter::node_t
::node_t()
  : deref(false)
  , name()
{
}


config_provider_sorter::node_t
::~node_t()
{
}

} // end namespace sprokit
