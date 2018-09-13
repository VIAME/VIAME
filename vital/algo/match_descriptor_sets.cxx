/*ckwg +29
 * Copyright 2017-2018 by Kitware, Inc.
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

/**
 * \file
 * \brief Implementation of bag of words matching
 */


#include <vital/algo/match_descriptor_sets.h>
#include <vital/algo/algorithm.txx>
#include <vital/exceptions/io.h>
#include <vital/vital_types.h>

#include <kwiversys/SystemTools.hxx>


namespace kwiver {
namespace vital {
namespace algo {

match_descriptor_sets
::match_descriptor_sets()
{
  attach_logger( "algo.match_descriptor_sets" );
}


std::vector<vital::frame_id_t>
match_descriptor_sets
::query_and_append(const vital::descriptor_set_sptr desc,
  frame_id_t frame)
{
  auto putative_matching_frames = this->query(desc);
  this->append_to_index(desc, frame);
  return putative_matching_frames;
}


} } } // end namespace

/// \cond DoxygenSuppress
INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::match_descriptor_sets);
/// \endcond
