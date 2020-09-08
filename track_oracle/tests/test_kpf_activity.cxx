/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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
 * \file
 * \brief Test reading KPF activities (and tracks)
 */

#include <gtest/gtest.h>
#include <test_gtest.h>

#include <track_oracle/core/track_oracle_core.h>
#include <track_oracle/core/track_field.h>
#include <track_oracle/data_terms/data_terms.h>
#include <track_oracle/file_formats/file_format_manager.h>
#include <track_oracle/file_formats/track_filter_kpf_activity/track_filter_kpf_activity.h>

#include <map>
#include <utility>

namespace to = ::kwiver::track_oracle;
namespace dt = ::kwiver::track_oracle::dt;
using std::string;

string g_data_dir;


// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  GET_ARG(1, g_data_dir);
  return RUN_ALL_TESTS();
}

// ------------------------------------------------------------------
TEST(track_oracle, kpf_activities)
{

  to::track_handle_list_type kpf_tracks, kpf_activities;
  {
    string fn = g_data_dir+"/test-large-IDs.geom.yml";
    bool rc = to::file_format_manager::read( fn, kpf_tracks );
    EXPECT_TRUE( rc ) << " reading tracks from '" << fn << "'";
    size_t n_read = kpf_tracks.size();
    EXPECT_EQ( n_read, 1 ) << " number of tracks read";
  }

  {
    const int domain=2;
    string fn = g_data_dir+"/test-large-IDs.activities.yml";
    bool rc = to::track_filter_kpf_activity::read( fn,
                                                   kpf_tracks,
                                                   domain,
                                                   kpf_activities );
    EXPECT_TRUE( rc ) << " reading activities from '" << fn << "'";
  }
}
