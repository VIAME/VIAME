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
 * @file   config_provider_sorter.h
 * @brief  Interface to config_provider_sorter class.
 */

#ifndef SPROKIT_PIPELINE_UTIL_CONFIG_PROVIDER_SORTER_H
#define SPROKIT_PIPELINE_UTIL_CONFIG_PROVIDER_SORTER_H

#include "bakery_base.h"

#include <vital/config/config_block.h>

#include <boost/graph/directed_graph.hpp>
#include <boost/variant.hpp>

#include <map>
#include <vector>


namespace sprokit {

// ----------------------------------------------------------------
/**
 * @brief
 *
 */
class config_provider_sorter
  : public boost::static_visitor<>
{
  public:
    config_provider_sorter();
    ~config_provider_sorter();

    void operator () (kwiver::vital::config_block_key_t const& key,
                      kwiver::vital::config_block_value_t const& value) const;
    void operator () (kwiver::vital::config_block_key_t const& key,
                      bakery_base::provider_request_t const& request);

    kwiver::vital::config_block_keys_t sorted() const;
  private:
    class node_t
    {
      public:
        node_t();
        ~node_t();

        bool deref;
        kwiver::vital::config_block_key_t name;
    };

    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, node_t> config_graph_t;
    typedef boost::graph_traits<config_graph_t>::vertex_descriptor vertex_t;
    typedef std::vector<vertex_t> vertices_t;
    typedef std::map<kwiver::vital::config_block_key_t, vertex_t> vertex_map_t;
    typedef vertex_map_t::value_type vertex_entry_t;

    vertex_map_t m_vertex_map;
    config_graph_t m_graph;
};

} // end namespace sprokit

#endif /* SPROKIT_PIPELINE_UTIL_CONFIG_PROVIDER_SORTER_H */
