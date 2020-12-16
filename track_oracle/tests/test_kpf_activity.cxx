// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
