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
 * \brief Test thread safety
 */

#include <string>
#include <vector>
#include <thread>
#include <sstream>

#include <gtest/gtest.h>
#include <test_gtest.h>

#include <vgl/vgl_area.h>
#include <vgl/vgl_box_2d.h>

#include <track_oracle/core/track_oracle_core.h>
#include <track_oracle/core/track_base.h>
#include <track_oracle/data_terms/data_terms.h>
#include <track_oracle/file_formats/file_format_manager.h>

namespace to = ::kwiver::track_oracle;

using std::string;
using std::vector;
using std::thread;
using std::ostringstream;

namespace { //anon

string g_data_dir;

//
// object to hold and compute some invariants about a set of tracks
//

struct track_stats
{
  size_t n_tracks;
  size_t n_frames;
  double sum_frame_area;
  track_stats(): n_tracks(0), n_frames(0), sum_frame_area(0.0) {}
  explicit track_stats( const to::track_handle_list_type& tracks );
  void set( const to::track_handle_list_type& tracks );
  void compare( const track_stats& other, const string& tag ) const;
};

track_stats
::track_stats( const to::track_handle_list_type& tracks )
{
  this->set( tracks );
}

void
track_stats
::set( const to::track_handle_list_type& tracks )
{
  this->n_tracks = tracks.size();
  this->n_frames = 0;
  this->sum_frame_area = 0.0;
  to::track_field< vgl_box_2d<double> > bounding_box( "bounding_box" );
  for (size_t i=0; i<this->n_tracks; ++i)
  {
    auto frames = to::track_oracle_core::get_frames( tracks[i] );
    this->n_frames += frames.size();
    for (size_t j=0; j<frames.size(); ++j)
    {
      if (bounding_box.exists( frames[j].row ))
      {
        this->sum_frame_area += vgl_area( bounding_box( frames[j].row ) );
      }
    }
  }
}

void
track_stats
::compare( const track_stats& other, const string& tag ) const
{
  EXPECT_EQ( this->n_tracks, other.n_tracks ) << ": " << tag;
  EXPECT_EQ( this->n_frames, other.n_frames ) << ": " << tag;
  EXPECT_EQ( this->sum_frame_area, other.sum_frame_area  ) << ": " << tag;
}

void
load_test_tracks( const string& tag, const string& fn, const track_stats& reference )
{
  to::track_handle_list_type tracks;
  bool rc = to::file_format_manager::read( fn, tracks );
  EXPECT_TRUE( rc ) << " reading for " << tag << " from '" << fn << "'";

  track_stats s( tracks );
  reference.compare( s, tag );
}

}; // ...anon



// ----------------------------------------------------------------------------
int
main( int argc, char* argv[] )
{
  ::testing::InitGoogleTest( &argc, argv );

#if GTEST_IS_THREADSAFE
  GET_ARG(1, g_data_dir);
  // ... do stuff
#endif
  return RUN_ALL_TESTS();
}

TEST( track_oracle, gtest_threadsafe )
{
#if GTEST_IS_THREADSAFE
  EXPECT_TRUE( true ) << "GTest is threadsafe";
#else
  EXPECT_TRUE( false ) << "GTest is not threadsafe";
#endif
}

TEST( track_oracle, track_oracle_threadsafe )
{

  string track_file = g_data_dir+"/generic_tracks.kw18";
  track_stats reference;
  {
    to::track_handle_list_type tracks;
    bool rc = to::file_format_manager::read( track_file, tracks );
    EXPECT_TRUE( rc ) << " reading from '" << track_file << "'";
    reference.set( tracks );
  }

  const size_t max_threads = 4; // arbitrary
  for (size_t n_threads = 2; n_threads < max_threads; ++n_threads )
  {
    vector< thread > threads;
    for (size_t i=0; i<n_threads; ++i)
    {
      ostringstream oss;
      oss << "Thread " << i+1 << " / " << n_threads;
      threads.push_back( thread( load_test_tracks, oss.str(), track_file, reference ));
    }
    for (auto& t: threads)
    {
      if (t.joinable())
      {
        t.join();
      }
    }
  } // ...for up to max_threads
}
