/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief GrabCut segmentation from a box or a partial labelling
///
/// What `cv::grabCut` did, ported from `imgproc/src/grabcut.cpp`,
/// `detail/gcgraph.hpp` and the part of `core/src/kmeans.cpp` it reaches.
/// OpenCV's copyright for those files:
///
///   Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
///   Copyright (C) 2009, Willow Garage Inc., all rights reserved.
///
///   Redistribution and use in source and binary forms, with or without
///   modification, are permitted provided that the following conditions are
///   met: redistributions of source code must retain the above copyright
///   notice, this list of conditions and the following disclaimer;
///   redistributions in binary form must reproduce the above copyright
///   notice, this list of conditions and the following disclaimer in the
///   documentation and/or other materials provided with the distribution;
///   neither the names of the copyright holders nor the names of the
///   contributors may be used to endorse or promote products derived from
///   this software without specific prior written permission.
///
///   This software is provided by the copyright holders and contributors
///   "as is" and any express or implied warranties are disclaimed.
///
/// The algorithm alternates two things. A pair of five-component Gaussian
/// mixtures models the colour of the background and of the foreground; a
/// minimum cut through a graph over the pixels then re-labels every
/// undecided pixel, with the mixtures setting how much each pixel wants to
/// join either side and the colour difference across each neighbouring pair
/// setting how much it costs to separate them. Each pass refits the mixtures
/// to the labelling the cut produced.
///
/// **`cv::grabCut` is not a function of its input.** Its mixtures are
/// initialised by k-means with `KMEANS_PP_CENTERS`, whose first centre and
/// whose three candidate draws per later centre come from `cv::theRNG()` --
/// a *global*, mutable generator. Anything else in the process that drew from
/// it first moves the answer, and not by a rounding: seeding it differently
/// moved 28 percent of one 90 by 120 mask. So this reproduces cv2's
/// generator exactly and starts it from the state a **fresh process** has,
/// `0xffffffff`, which makes the result a function of the input alone and
/// equal to what cv2 gives before anything else has used the generator.
///
/// What else changed in the port, and nothing more did:
///
/// - `cv::parallel_for_` became plain loops. Every one of them was a pure
///   map, so the order never mattered.
/// - The max-flow's intrusive linked list of active vertices became indices
///   rather than pointers, with the sentinel spelled out. Its traversal
///   order is load-bearing -- the cut is only unique up to ties, and the
///   order the trees grow and the order orphans are adopted decide which
///   tie is taken -- so the structure is transcribed rather than tidied,
///   including the unused first edge pair that lets `ei ^ 1` mean "the
///   reverse edge" and `ei == 0` mean "no edge".
/// - OpenCL is dropped, as elsewhere.

#ifndef VIAME_IMAGE_KERNELS_GRABCUT_H
#define VIAME_IMAGE_KERNELS_GRABCUT_H

#include <viame/core_types/image.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace viame {
namespace image_kernels {

/// The four labels a GrabCut mask carries, with `cv::GrabCutClasses`' values.
enum class grabcut_label : uint8_t
{
  BACKGROUND = 0,
  FOREGROUND = 1,
  PROBABLY_BACKGROUND = 2,
  PROBABLY_FOREGROUND = 3,
};

/// How `grab_cut` starts.
enum class grabcut_mode
{
  /// `GC_INIT_WITH_RECT`: the mask is built from the rectangle, everything
  /// inside it probably foreground and everything outside it background.
  WITH_RECT,
  /// `GC_INIT_WITH_MASK`: the mask is taken as given and the mixtures are
  /// initialised from it.
  WITH_MASK,
  /// `GC_EVAL`: the mask is taken as given and the mixtures are whatever the
  /// caller's models hold, which for a fresh pair of models is nothing.
  EVAL,
  /// `GC_EVAL_FREEZE_MODEL`: one pass, and the mixtures are not refitted.
  EVAL_FREEZE_MODEL,
};

namespace detail {

// ----------------------------------------------------------------------------
/// `cv::RNG`: the multiply-with-carry generator, and the two conversions
/// k-means++ draws through.
///
/// The state is 64 bits and only the low 32 are returned, so `double()`
/// consumes **two** draws where `unsigned()` consumes one -- which is why the
/// sequence cannot be reconstructed from the values alone. The default state
/// is what a fresh `cv::RNG` and therefore a fresh `cv::theRNG()` holds.
class mwc_rng
{
public:
  explicit mwc_rng( uint64_t state = 0xffffffffu )
    : state_( state ? state : 0xffffffffu ) {}

  uint32_t next()
  {
    state_ = static_cast< uint64_t >( static_cast< uint32_t >( state_ ) ) *
      4164903690u + static_cast< uint32_t >( state_ >> 32 );
    return static_cast< uint32_t >( state_ );
  }

  uint32_t unsigned_value() { return next(); }

  double double_value()
  {
    auto const high = static_cast< uint64_t >( next() );
    return static_cast< double >( ( high << 32 ) | next() ) *
      5.4210108624275221700372640043497e-20;
  }

private:
  uint64_t state_;
};

// ----------------------------------------------------------------------------
/// `hal::normL2Sqr_` over three floats: a **float32** accumulation.
///
/// Three is short enough that OpenCV's vector body never runs, so this is its
/// scalar tail. In float rather than double because that is what it returns
/// and what k-means then compares.
inline float
squared_distance( float const* a, float const* b )
{
  float total = 0.0f;
  for( int j = 0; j < 3; ++j )
  {
    float const t = a[j] - b[j];
    total += t * t;
  }
  return total;
}

// ----------------------------------------------------------------------------
/// `generateCentersPP`: k-means++ seeding, three candidates per centre.
///
/// The first centre is a single draw; each later one takes three candidates
/// sampled proportionally to the current squared distance, keeps whichever
/// lowers the total most, and carries that total forward. The running total
/// is a **double** sum of float distances, and the candidate search walks the
/// same cumulative array OpenCV walks, stopping one short of the end.
inline std::vector< int >
kmeans_pp_centers( std::vector< float > const& data, int count, int k,
                   mwc_rng& rng )
{
  constexpr int trials = 3;

  std::vector< int > centers( static_cast< size_t >( k ), 0 );
  std::vector< float > distance( static_cast< size_t >( count ) );
  std::vector< float > candidate( static_cast< size_t >( count ) );
  std::vector< float > best_distance( static_cast< size_t >( count ) );

  centers[0] = static_cast< int >(
    rng.unsigned_value() % static_cast< uint32_t >( count ) );

  double total = 0.0;
  for( int i = 0; i < count; ++i )
  {
    distance[i] = squared_distance( &data[ 3 * i ],
                                    &data[ 3 * centers[0] ] );
    total += distance[i];
  }

  for( int c = 1; c < k; ++c )
  {
    double best_total = DBL_MAX;
    int best_center = -1;

    for( int j = 0; j < trials; ++j )
    {
      auto p = rng.double_value() * total;
      int pick = 0;
      for( ; pick < count - 1; ++pick )
      {
        p -= distance[pick];
        if( p <= 0 ) { break; }
      }

      double sum = 0.0;
      for( int i = 0; i < count; ++i )
      {
        candidate[i] = std::min(
          distance[i], squared_distance( &data[ 3 * i ], &data[ 3 * pick ] ) );
        sum += candidate[i];
      }

      if( sum < best_total )
      {
        best_total = sum;
        best_center = pick;
        best_distance.swap( candidate );
      }
    }

    if( best_center < 0 )
    {
      throw std::runtime_error(
        "grab_cut: k-means cannot place a centre; the samples hold a huge or "
        "not-a-number value" );
    }

    centers[c] = best_center;
    total = best_total;
    distance.swap( best_distance );
  }

  return centers;
}

// ----------------------------------------------------------------------------
/// `cv::kmeans` with `KMEANS_PP_CENTERS`, one attempt and an iteration limit.
///
/// The parts that look incidental and are not: the epsilon is `FLT_EPSILON`
/// **squared** when the criteria ask only for a count; the last iteration
/// does not re-assign labels, so the labels returned belong to the
/// second-to-last centres; and an empty cluster is refilled by moving the
/// single farthest point out of the largest cluster rather than by re-seeding.
inline std::vector< int >
kmeans_pp( std::vector< float > const& data, int count, int k,
           mwc_rng& rng, int max_count )
{
  std::vector< int > labels( static_cast< size_t >( count ), -1 );
  std::vector< float > centers( static_cast< size_t >( 3 * k ), 0.0f );
  std::vector< float > previous( static_cast< size_t >( 3 * k ), 0.0f );
  std::vector< int > counters( static_cast< size_t >( k ), 0 );

  auto const epsilon =
    static_cast< double >( FLT_EPSILON ) * static_cast< double >( FLT_EPSILON );
  max_count = std::min( std::max( max_count, 2 ), 100 );

  for( int iteration = 0;; )
  {
    auto shift = iteration == 0 ? DBL_MAX : 0.0;
    centers.swap( previous );

    if( iteration == 0 )
    {
      auto const chosen = kmeans_pp_centers( data, count, k, rng );
      for( int c = 0; c < k; ++c )
      {
        for( int j = 0; j < 3; ++j )
        { centers[ 3 * c + j ] = data[ 3 * chosen[c] + j ]; }
      }
    }
    else
    {
      std::fill( centers.begin(), centers.end(), 0.0f );
      std::fill( counters.begin(), counters.end(), 0 );

      for( int i = 0; i < count; ++i )
      {
        auto const c = labels[i];
        for( int j = 0; j < 3; ++j )
        { centers[ 3 * c + j ] += data[ 3 * i + j ]; }
        ++counters[c];
      }

      for( int c = 0; c < k; ++c )
      {
        if( counters[c] != 0 ) { continue; }

        auto biggest = 0;
        for( int other = 1; other < k; ++other )
        {
          if( counters[biggest] < counters[other] ) { biggest = other; }
        }

        float normalised[3];
        auto const scale = 1.0f / static_cast< float >( counters[biggest] );
        for( int j = 0; j < 3; ++j )
        { normalised[j] = centers[ 3 * biggest + j ] * scale; }

        double farthest_distance = 0.0;
        int farthest = -1;
        for( int i = 0; i < count; ++i )
        {
          if( labels[i] != biggest ) { continue; }
          auto const d = static_cast< double >(
            squared_distance( &data[ 3 * i ], normalised ) );
          // `<=`, so the **last** farthest point wins a tie.
          if( farthest_distance <= d )
          {
            farthest_distance = d;
            farthest = i;
          }
        }

        --counters[biggest];
        ++counters[c];
        labels[farthest] = c;
        for( int j = 0; j < 3; ++j )
        {
          centers[ 3 * biggest + j ] -= data[ 3 * farthest + j ];
          centers[ 3 * c + j ] += data[ 3 * farthest + j ];
        }
      }

      for( int c = 0; c < k; ++c )
      {
        auto const scale = 1.0f / static_cast< float >( counters[c] );
        for( int j = 0; j < 3; ++j ) { centers[ 3 * c + j ] *= scale; }

        if( iteration > 0 )
        {
          double moved = 0.0;
          for( int j = 0; j < 3; ++j )
          {
            auto const t = static_cast< double >( centers[ 3 * c + j ] ) -
                           static_cast< double >( previous[ 3 * c + j ] );
            moved += t * t;
          }
          shift = std::max( shift, moved );
        }
      }
    }

    ++iteration;
    if( iteration == std::max( max_count, 2 ) || shift <= epsilon ) { break; }

    for( int i = 0; i < count; ++i )
    {
      double best = DBL_MAX;
      int chosen = 0;
      for( int c = 0; c < k; ++c )
      {
        auto const d = static_cast< double >(
          squared_distance( &data[ 3 * i ], &centers[ 3 * c ] ) );
        if( best > d )
        {
          best = d;
          chosen = c;
        }
      }
      labels[i] = chosen;
    }
  }

  return labels;
}

// ----------------------------------------------------------------------------
/// One side's five-component Gaussian mixture over colour.
///
/// The model is 65 doubles, laid out as `cv::grabCut`'s is -- five weights,
/// then five means of three, then five covariances of nine -- because a
/// caller carries it between frames and may have one cv2 wrote.
///
/// Two details are worth stating. A covariance whose determinant falls to
/// 1e-6 or below has 0.01 added to its diagonal, which is the only guard
/// against a component collapsing onto one colour; and `which_component`
/// starts its search at zero **with a threshold of zero**, so a colour that
/// every component gives probability zero lands in component zero rather
/// than nowhere.
class colour_mixture
{
public:
  static constexpr int components = 5;
  static constexpr int model_size = 13 * components;

  explicit colour_mixture( std::vector< double >& model )
    : model_( model )
  {
    if( model_.empty() )
    { model_.assign( static_cast< size_t >( model_size ), 0.0 ); }
    else if( model_.size() != static_cast< size_t >( model_size ) )
    {
      throw std::invalid_argument(
        "grab_cut: a mixture model has to hold 65 doubles" );
    }

    for( int c = 0; c < components; ++c )
    {
      if( weight( c ) > 0 ) { invert( c, 0.0 ); }
    }
  }

  double& weight( int c ) { return model_[ static_cast< size_t >( c ) ]; }
  double weight( int c ) const { return model_[ static_cast< size_t >( c ) ]; }

  double* mean( int c )
  { return model_.data() + components + 3 * c; }
  double const* mean( int c ) const
  { return model_.data() + components + 3 * c; }

  double* covariance( int c )
  { return model_.data() + components + 3 * components + 9 * c; }
  double const* covariance( int c ) const
  { return model_.data() + components + 3 * components + 9 * c; }

  /// How much component \p c likes this colour, unweighted.
  double
  likelihood( int c, double const( &colour )[3] ) const
  {
    if( weight( c ) <= 0 ) { return 0.0; }

    auto const* m = mean( c );
    double const d[3] = { colour[0] - m[0], colour[1] - m[1],
                          colour[2] - m[2] };
    auto const& iv = inverse_[ static_cast< size_t >( c ) ];
    auto const mult =
      d[0] * ( d[0] * iv[0][0] + d[1] * iv[1][0] + d[2] * iv[2][0] ) +
      d[1] * ( d[0] * iv[0][1] + d[1] * iv[1][1] + d[2] * iv[2][1] ) +
      d[2] * ( d[0] * iv[0][2] + d[1] * iv[1][2] + d[2] * iv[2][2] );
    return 1.0 / std::sqrt( determinant_[ static_cast< size_t >( c ) ] ) *
           std::exp( -0.5 * mult );
  }

  /// How much the mixture likes this colour.
  double
  likelihood( double const( &colour )[3] ) const
  {
    double total = 0.0;
    for( int c = 0; c < components; ++c )
    { total += weight( c ) * likelihood( c, colour ); }
    return total;
  }

  int
  which_component( double const( &colour )[3] ) const
  {
    int chosen = 0;
    double best = 0.0;
    for( int c = 0; c < components; ++c )
    {
      auto const p = likelihood( c, colour );
      if( p > best )
      {
        chosen = c;
        best = p;
      }
    }
    return chosen;
  }

  void
  begin_learning()
  {
    sums_.assign( static_cast< size_t >( 3 * components ), 0.0 );
    products_.assign( static_cast< size_t >( 9 * components ), 0.0 );
    counts_.assign( static_cast< size_t >( components ), 0 );
    total_ = 0;
  }

  void
  add_sample( int c, double const( &colour )[3] )
  {
    auto* s = sums_.data() + 3 * c;
    auto* p = products_.data() + 9 * c;
    for( int i = 0; i < 3; ++i )
    {
      s[i] += colour[i];
      for( int j = 0; j < 3; ++j ) { p[ 3 * i + j ] += colour[i] * colour[j]; }
    }
    ++counts_[ static_cast< size_t >( c ) ];
    ++total_;
  }

  void
  end_learning()
  {
    for( int c = 0; c < components; ++c )
    {
      auto const n = counts_[ static_cast< size_t >( c ) ];
      if( n == 0 )
      {
        weight( c ) = 0;
        continue;
      }

      auto const inverse_n = 1.0 / static_cast< double >( n );
      weight( c ) =
        static_cast< double >( n ) / static_cast< double >( total_ );

      auto* m = mean( c );
      auto const* s = sums_.data() + 3 * c;
      for( int i = 0; i < 3; ++i ) { m[i] = s[i] * inverse_n; }

      auto* cov = covariance( c );
      auto const* p = products_.data() + 9 * c;
      for( int i = 0; i < 3; ++i )
      {
        for( int j = 0; j < 3; ++j )
        { cov[ 3 * i + j ] = p[ 3 * i + j ] * inverse_n - m[i] * m[j]; }
      }

      invert( c, 0.01 );
    }
  }

private:
  void
  invert( int c, double singular_fix )
  {
    if( weight( c ) <= 0 ) { return; }

    auto* v = covariance( c );
    auto const compute = [ v ]()
    {
      return v[0] * ( v[4] * v[8] - v[5] * v[7] ) -
             v[1] * ( v[3] * v[8] - v[5] * v[6] ) +
             v[2] * ( v[3] * v[7] - v[4] * v[6] );
    };

    auto determinant = compute();
    if( determinant <= 1e-6 && singular_fix > 0 )
    {
      // White noise on the diagonal, which is the only thing keeping a
      // component that has collapsed onto one colour invertible.
      v[0] += singular_fix;
      v[4] += singular_fix;
      v[8] += singular_fix;
      determinant = compute();
    }

    determinant_[ static_cast< size_t >( c ) ] = determinant;

    if( !( determinant > DBL_EPSILON ) )
    {
      throw std::runtime_error(
        "grab_cut: a mixture component has a singular covariance" );
    }

    auto const inverse = 1.0 / determinant;
    auto& iv = inverse_[ static_cast< size_t >( c ) ];
    iv[0][0] = ( v[4] * v[8] - v[5] * v[7] ) * inverse;
    iv[1][0] = -( v[3] * v[8] - v[5] * v[6] ) * inverse;
    iv[2][0] = ( v[3] * v[7] - v[4] * v[6] ) * inverse;
    iv[0][1] = -( v[1] * v[8] - v[2] * v[7] ) * inverse;
    iv[1][1] = ( v[0] * v[8] - v[2] * v[6] ) * inverse;
    iv[2][1] = -( v[0] * v[7] - v[1] * v[6] ) * inverse;
    iv[0][2] = ( v[1] * v[5] - v[2] * v[4] ) * inverse;
    iv[1][2] = -( v[0] * v[5] - v[2] * v[3] ) * inverse;
    iv[2][2] = ( v[0] * v[4] - v[1] * v[3] ) * inverse;
  }

  std::vector< double >& model_;
  double inverse_[components][3][3] = {};
  double determinant_[components] = {};

  std::vector< double > sums_;
  std::vector< double > products_;
  std::vector< long > counts_;
  long total_ = 0;
};

// ----------------------------------------------------------------------------
/// `cv::detail::GCGraph`: Boykov-Kolmogorov maximum flow over doubles.
///
/// Two search trees grow, one from the source and one from the sink, until an
/// edge joins them; the path through that edge is saturated; the vertices
/// whose parent edge the saturation emptied become orphans and are given new
/// parents, or pushed back into the active list if they have none. The cut is
/// what the trees hold when no joining edge is left.
///
/// **The traversal order is part of the answer.** A minimum cut is unique
/// only up to ties, and which tie this lands on follows from the order the
/// active list is drained, the order each vertex's edges are visited -- most
/// recently added first, since `add_edges` pushes onto the front of a
/// per-vertex chain -- and the orphan list being a **stack**. So this is
/// transcribed rather than restructured, down to the unused first edge pair
/// that lets `edge ^ 1` mean the reverse edge and `edge == 0` mean none.
class flow_graph
{
public:
  static constexpr int terminal = -1;
  static constexpr int orphan = -2;
  /// Not in the active list at all, as against `nil`, which is its end.
  static constexpr int detached = -3;
  static constexpr int nil = -1;

  void
  reserve( size_t vertices, size_t edges )
  {
    parent_.reserve( vertices );
    first_.reserve( vertices );
    stamp_.reserve( vertices );
    distance_.reserve( vertices );
    weight_.reserve( vertices );
    side_.reserve( vertices );
    next_.reserve( vertices );
    edge_target_.reserve( edges + 2 );
    edge_next_.reserve( edges + 2 );
    edge_weight_.reserve( edges + 2 );
    // The wasted first pair, so that index zero can mean "no edge".
    edge_target_.assign( 2, 0 );
    edge_next_.assign( 2, 0 );
    edge_weight_.assign( 2, 0.0 );
  }

  int
  add_vertex()
  {
    parent_.push_back( 0 );
    first_.push_back( 0 );
    stamp_.push_back( 0 );
    distance_.push_back( 0 );
    weight_.push_back( 0.0 );
    side_.push_back( 0 );
    next_.push_back( detached );
    return static_cast< int >( parent_.size() ) - 1;
  }

  void
  add_edges( int i, int j, double forward, double backward )
  {
    edge_target_.push_back( j );
    edge_next_.push_back( first_[ static_cast< size_t >( i ) ] );
    edge_weight_.push_back( forward );
    first_[ static_cast< size_t >( i ) ] =
      static_cast< int >( edge_target_.size() ) - 1;

    edge_target_.push_back( i );
    edge_next_.push_back( first_[ static_cast< size_t >( j ) ] );
    edge_weight_.push_back( backward );
    first_[ static_cast< size_t >( j ) ] =
      static_cast< int >( edge_target_.size() ) - 1;
  }

  void
  add_terminal_weights( int i, double from_source, double to_sink )
  {
    auto const held = weight_[ static_cast< size_t >( i ) ];
    if( held > 0 ) { from_source += held; }
    else { to_sink -= held; }
    flow_ += from_source < to_sink ? from_source : to_sink;
    weight_[ static_cast< size_t >( i ) ] = from_source - to_sink;
  }

  bool
  in_source_segment( int i ) const
  { return side_[ static_cast< size_t >( i ) ] == 0; }

  double max_flow();

private:
  std::vector< int > parent_;
  std::vector< int > first_;
  std::vector< int > stamp_;
  std::vector< int > distance_;
  std::vector< double > weight_;
  std::vector< uint8_t > side_;
  std::vector< int > next_;

  std::vector< int > edge_target_;
  std::vector< int > edge_next_;
  std::vector< double > edge_weight_;

  double flow_ = 0.0;
  /// The sentinel vertex's `next`, which the real vertices' list hangs off.
  int stub_next_ = nil;
};

// ----------------------------------------------------------------------------
inline double
flow_graph::max_flow()
{
  if( parent_.empty() || edge_target_.size() <= 2 )
  {
    throw std::invalid_argument( "grab_cut: the flow graph has no shape" );
  }

  auto const count = static_cast< int >( parent_.size() );
  auto const following = [&]( int v ) -> int&
  { return v == nil ? stub_next_ : next_[ static_cast< size_t >( v ) ]; };

  int last = nil;
  for( int i = 0; i < count; ++i )
  {
    stamp_[ static_cast< size_t >( i ) ] = 0;
    if( weight_[ static_cast< size_t >( i ) ] != 0 )
    {
      following( last ) = i;
      last = i;
      distance_[ static_cast< size_t >( i ) ] = 1;
      parent_[ static_cast< size_t >( i ) ] = terminal;
      side_[ static_cast< size_t >( i ) ] =
        weight_[ static_cast< size_t >( i ) ] < 0 ? 1 : 0;
    }
    else { parent_[ static_cast< size_t >( i ) ] = 0; }
  }
  auto active = following( nil );
  following( last ) = nil;
  stub_next_ = detached;

  std::vector< int > orphans;
  int stamp = 0;

  for( ;; )
  {
    int joining = -1;
    int edge = 0;
    int v = nil;

    // Grow the two trees until an edge joins them.
    while( active != nil )
    {
      v = active;
      if( parent_[ static_cast< size_t >( v ) ] )
      {
        auto const side = side_[ static_cast< size_t >( v ) ];
        for( edge = first_[ static_cast< size_t >( v ) ]; edge != 0;
             edge = edge_next_[ static_cast< size_t >( edge ) ] )
        {
          if( edge_weight_[ static_cast< size_t >( edge ^ side ) ] == 0 )
          { continue; }

          auto const u = edge_target_[ static_cast< size_t >( edge ) ];
          if( !parent_[ static_cast< size_t >( u ) ] )
          {
            side_[ static_cast< size_t >( u ) ] = side;
            parent_[ static_cast< size_t >( u ) ] = edge ^ 1;
            stamp_[ static_cast< size_t >( u ) ] =
              stamp_[ static_cast< size_t >( v ) ];
            distance_[ static_cast< size_t >( u ) ] =
              distance_[ static_cast< size_t >( v ) ] + 1;
            if( next_[ static_cast< size_t >( u ) ] == detached )
            {
              next_[ static_cast< size_t >( u ) ] = nil;
              following( last ) = u;
              last = u;
            }
            continue;
          }

          if( side_[ static_cast< size_t >( u ) ] != side )
          {
            joining = edge ^ side;
            break;
          }

          if( distance_[ static_cast< size_t >( u ) ] >
                distance_[ static_cast< size_t >( v ) ] + 1 &&
              stamp_[ static_cast< size_t >( u ) ] <=
                stamp_[ static_cast< size_t >( v ) ] )
          {
            parent_[ static_cast< size_t >( u ) ] = edge ^ 1;
            stamp_[ static_cast< size_t >( u ) ] =
              stamp_[ static_cast< size_t >( v ) ];
            distance_[ static_cast< size_t >( u ) ] =
              distance_[ static_cast< size_t >( v ) ] + 1;
          }
        }
        if( joining > 0 ) { break; }
      }
      active = following( active );
      next_[ static_cast< size_t >( v ) ] = detached;
    }

    if( joining <= 0 ) { break; }

    // The smallest capacity along the path through the joining edge.
    auto smallest = edge_weight_[ static_cast< size_t >( joining ) ];
    for( int k = 1; k >= 0; --k )
    {
      v = edge_target_[ static_cast< size_t >( joining ^ k ) ];
      for( ;; )
      {
        edge = parent_[ static_cast< size_t >( v ) ];
        if( edge < 0 ) { break; }
        smallest = std::min(
          smallest, edge_weight_[ static_cast< size_t >( edge ^ k ) ] );
        v = edge_target_[ static_cast< size_t >( edge ) ];
      }
      smallest = std::min(
        smallest, std::fabs( weight_[ static_cast< size_t >( v ) ] ) );
    }

    edge_weight_[ static_cast< size_t >( joining ) ] -= smallest;
    edge_weight_[ static_cast< size_t >( joining ^ 1 ) ] += smallest;
    flow_ += smallest;

    // Saturate the path and collect whatever it orphaned.
    for( int k = 1; k >= 0; --k )
    {
      v = edge_target_[ static_cast< size_t >( joining ^ k ) ];
      for( ;; )
      {
        edge = parent_[ static_cast< size_t >( v ) ];
        if( edge < 0 ) { break; }
        edge_weight_[ static_cast< size_t >( edge ^ ( k ^ 1 ) ) ] += smallest;
        edge_weight_[ static_cast< size_t >( edge ^ k ) ] -= smallest;
        if( edge_weight_[ static_cast< size_t >( edge ^ k ) ] == 0 )
        {
          orphans.push_back( v );
          parent_[ static_cast< size_t >( v ) ] = orphan;
        }
        v = edge_target_[ static_cast< size_t >( edge ) ];
      }

      weight_[ static_cast< size_t >( v ) ] += smallest * ( 1 - k * 2 );
      if( weight_[ static_cast< size_t >( v ) ] == 0 )
      {
        orphans.push_back( v );
        parent_[ static_cast< size_t >( v ) ] = orphan;
      }
    }

    // Adopt the orphans, newest first.
    ++stamp;
    while( !orphans.empty() )
    {
      auto const child = orphans.back();
      orphans.pop_back();

      auto shortest = INT_MAX;
      joining = 0;
      auto const side = side_[ static_cast< size_t >( child ) ];

      for( edge = first_[ static_cast< size_t >( child ) ]; edge != 0;
           edge = edge_next_[ static_cast< size_t >( edge ) ] )
      {
        if( edge_weight_[ static_cast< size_t >( edge ^ ( side ^ 1 ) ) ] == 0 )
        { continue; }
        auto u = edge_target_[ static_cast< size_t >( edge ) ];
        if( side_[ static_cast< size_t >( u ) ] != side ||
            parent_[ static_cast< size_t >( u ) ] == 0 )
        { continue; }

        // How far this candidate parent is from its tree's root.
        int depth = 0;
        for( ;; )
        {
          if( stamp_[ static_cast< size_t >( u ) ] == stamp )
          {
            depth += distance_[ static_cast< size_t >( u ) ];
            break;
          }
          auto const up = parent_[ static_cast< size_t >( u ) ];
          ++depth;
          if( up < 0 )
          {
            if( up == orphan ) { depth = INT_MAX - 1; }
            else
            {
              stamp_[ static_cast< size_t >( u ) ] = stamp;
              distance_[ static_cast< size_t >( u ) ] = 1;
            }
            break;
          }
          u = edge_target_[ static_cast< size_t >( up ) ];
        }

        if( ++depth < INT_MAX )
        {
          if( depth < shortest )
          {
            shortest = depth;
            joining = edge;
          }
          for( u = edge_target_[ static_cast< size_t >( edge ) ];
               stamp_[ static_cast< size_t >( u ) ] != stamp;
               u = edge_target_[ static_cast< size_t >(
                     parent_[ static_cast< size_t >( u ) ] ) ] )
          {
            stamp_[ static_cast< size_t >( u ) ] = stamp;
            distance_[ static_cast< size_t >( u ) ] = --depth;
          }
        }
      }

      parent_[ static_cast< size_t >( child ) ] = joining;
      if( joining > 0 )
      {
        stamp_[ static_cast< size_t >( child ) ] = stamp;
        distance_[ static_cast< size_t >( child ) ] = shortest;
        continue;
      }

      // No parent: wake the neighbours that could become one, and orphan
      // anything that was depending on this vertex.
      stamp_[ static_cast< size_t >( child ) ] = 0;
      for( edge = first_[ static_cast< size_t >( child ) ]; edge != 0;
           edge = edge_next_[ static_cast< size_t >( edge ) ] )
      {
        auto const u = edge_target_[ static_cast< size_t >( edge ) ];
        auto const up = parent_[ static_cast< size_t >( u ) ];
        if( side_[ static_cast< size_t >( u ) ] != side || !up ) { continue; }
        if( edge_weight_[ static_cast< size_t >( edge ^ ( side ^ 1 ) ) ] != 0 &&
            next_[ static_cast< size_t >( u ) ] == detached )
        {
          next_[ static_cast< size_t >( u ) ] = nil;
          following( last ) = u;
          last = u;
        }
        if( up > 0 &&
            edge_target_[ static_cast< size_t >( up ) ] == child )
        {
          orphans.push_back( u );
          parent_[ static_cast< size_t >( u ) ] = orphan;
        }
      }
    }
  }

  return flow_;
}

// ----------------------------------------------------------------------------
/// `calcBeta`: the reciprocal of twice the mean squared colour difference
/// between neighbours, which is what turns a colour distance into a cost.
///
/// The sum runs over the four neighbours behind each pixel -- left, up-left,
/// up and up-right -- so every unordered pair is counted once, and the
/// divisor is the number of such pairs rather than the pixel count.
inline double
grabcut_beta( viame::image_of< uint8_t > const& image )
{
  auto const width = static_cast< long >( image.width() );
  auto const height = static_cast< long >( image.height() );

  auto const at = [&]( long y, long x, int plane ) -> double
  {
    return image( static_cast< size_t >( x ), static_cast< size_t >( y ),
                  static_cast< size_t >( plane ) );
  };
  auto const difference = [&]( long y, long x, long oy, long ox ) -> double
  {
    double total = 0.0;
    for( int p = 0; p < 3; ++p )
    {
      auto const d = at( y, x, p ) - at( oy, ox, p );
      total += d * d;
    }
    return total;
  };

  double beta = 0.0;
  for( long y = 0; y < height; ++y )
  {
    for( long x = 0; x < width; ++x )
    {
      if( x > 0 ) { beta += difference( y, x, y, x - 1 ); }
      if( y > 0 && x > 0 ) { beta += difference( y, x, y - 1, x - 1 ); }
      if( y > 0 ) { beta += difference( y, x, y - 1, x ); }
      if( y > 0 && x < width - 1 )
      { beta += difference( y, x, y - 1, x + 1 ); }
    }
  }

  if( beta <= DBL_EPSILON ) { return 0.0; }

  auto const pairs = static_cast< double >(
    4 * width * height - 3 * width - 3 * height + 2 );
  return 1.0 / ( 2 * beta / pairs );
}

// ----------------------------------------------------------------------------
/// The cost of separating each pixel from its four neighbours behind it.
///
/// `gamma` for the two axis-aligned directions and `gamma / sqrt(2)` for the
/// two diagonals, each damped by how different the two colours are. The
/// square root is of a **float** literal in OpenCV, and the difference from
/// the double one reaches the answer.
struct grabcut_weights
{
  std::vector< double > left, up_left, up, up_right;
};

inline grabcut_weights
grabcut_neighbour_weights( viame::image_of< uint8_t > const& image,
                           double beta, double gamma )
{
  auto const width = static_cast< long >( image.width() );
  auto const height = static_cast< long >( image.height() );
  auto const count = static_cast< size_t >( width * height );
  auto const diagonal =
    gamma / std::sqrt( static_cast< double >( 2.0f ) );

  grabcut_weights out;
  out.left.assign( count, 0.0 );
  out.up_left.assign( count, 0.0 );
  out.up.assign( count, 0.0 );
  out.up_right.assign( count, 0.0 );

  auto const at = [&]( long y, long x, int plane ) -> double
  {
    return image( static_cast< size_t >( x ), static_cast< size_t >( y ),
                  static_cast< size_t >( plane ) );
  };
  auto const damped = [&]( long y, long x, long oy, long ox, double scale )
  {
    double total = 0.0;
    for( int p = 0; p < 3; ++p )
    {
      auto const d = at( y, x, p ) - at( oy, ox, p );
      total += d * d;
    }
    return scale * std::exp( -beta * total );
  };

  for( long y = 0; y < height; ++y )
  {
    for( long x = 0; x < width; ++x )
    {
      auto const i = static_cast< size_t >( y * width + x );
      if( x - 1 >= 0 ) { out.left[i] = damped( y, x, y, x - 1, gamma ); }
      if( x - 1 >= 0 && y - 1 >= 0 )
      { out.up_left[i] = damped( y, x, y - 1, x - 1, diagonal ); }
      if( y - 1 >= 0 ) { out.up[i] = damped( y, x, y - 1, x, gamma ); }
      if( x + 1 < width && y - 1 >= 0 )
      { out.up_right[i] = damped( y, x, y - 1, x + 1, diagonal ); }
    }
  }

  return out;
}

} // namespace detail

// ----------------------------------------------------------------------------
/// `cv::grabCut`: segment an object from a box or from a partial labelling.
///
/// \p mask is read and written. On `WITH_RECT` it is replaced wholesale --
/// background outside the rectangle, probably foreground inside it -- and on
/// every other mode it has to hold only the four `grabcut_label` values. What
/// comes back has each *probable* label resolved to one side or the other;
/// the pixels the caller marked definitely background or definitely
/// foreground are left exactly as they were, which is what makes those marks
/// a constraint rather than a hint.
///
/// \p background_model and \p foreground_model are the caller's, 65 doubles
/// each, and are carried across calls so that a second call can refine the
/// first. An empty vector is filled in.
///
/// The three planes may be in any order. A permutation of the colour axes
/// permutes each mixture's mean and conjugates its covariance, which leaves
/// the Mahalanobis distance and the determinant alone, and the colour
/// differences are sums of exactly representable integers -- so RGB and BGR
/// give the same mask rather than nearly the same one.
///
/// @param image three planes of bytes
/// @param mask the labelling, read and written
/// @param rect x, y, width, height; used by `WITH_RECT` alone
/// @param iterations how many times to refit and re-cut
/// @param mode how to start
/// @param rng_state what `cv::theRNG()` holds when the k-means seeding
///   starts. The default is the state a **fresh process** has, which is what
///   makes this a function of its input; passing another value reproduces
///   what cv2 gives after `cv2.setRNGSeed` of the same number, which is the
///   only way to compare the two at more than one point.
inline void
grab_cut( viame::image_of< uint8_t > const& image,
          viame::image_of< uint8_t >& mask,
          std::array< int, 4 > const& rect,
          int iterations,
          grabcut_mode mode,
          std::vector< double >& background_model,
          std::vector< double >& foreground_model,
          uint64_t rng_state = 0xffffffffu )
{
  using namespace detail;

  if( image.depth() != 3 )
  {
    throw std::invalid_argument( "grab_cut wants a three plane image" );
  }

  if( image.width() == 0 || image.height() == 0 )
  {
    throw std::invalid_argument( "grab_cut: the image has no area" );
  }

  auto const width = static_cast< long >( image.width() );
  auto const height = static_cast< long >( image.height() );

  auto const is_background = []( uint8_t v )
  {
    return v == static_cast< uint8_t >( grabcut_label::BACKGROUND ) ||
           v == static_cast< uint8_t >( grabcut_label::PROBABLY_BACKGROUND );
  };
  auto const is_probable = []( uint8_t v )
  {
    return v == static_cast< uint8_t >( grabcut_label::PROBABLY_BACKGROUND ) ||
           v == static_cast< uint8_t >( grabcut_label::PROBABLY_FOREGROUND );
  };

  if( mode == grabcut_mode::WITH_RECT )
  {
    mask = viame::image_of< uint8_t >( image.width(), image.height(), 1 );
    for( long y = 0; y < height; ++y )
    {
      for( long x = 0; x < width; ++x )
      {
        mask( static_cast< size_t >( x ), static_cast< size_t >( y ) ) =
          static_cast< uint8_t >( grabcut_label::BACKGROUND );
      }
    }

    auto const x0 = std::max( 0, rect[0] );
    auto const y0 = std::max( 0, rect[1] );
    auto const x1 = x0 + std::min( rect[2], static_cast< int >( width ) - x0 );
    auto const y1 = y0 + std::min( rect[3], static_cast< int >( height ) - y0 );
    for( int y = y0; y < y1; ++y )
    {
      for( int x = x0; x < x1; ++x )
      {
        mask( static_cast< size_t >( x ), static_cast< size_t >( y ) ) =
          static_cast< uint8_t >( grabcut_label::PROBABLY_FOREGROUND );
      }
    }
  }
  else
  {
    if( mask.width() != image.width() || mask.height() != image.height() ||
        mask.depth() != 1 )
    {
      throw std::invalid_argument(
        "grab_cut: the mask has to be one plane the size of the image" );
    }
    for( long y = 0; y < height; ++y )
    {
      for( long x = 0; x < width; ++x )
      {
        auto const v =
          mask( static_cast< size_t >( x ), static_cast< size_t >( y ) );
        if( v > 3 )
        {
          throw std::invalid_argument(
            "grab_cut: a mask value is not one of the four labels" );
        }
      }
    }
  }

  colour_mixture background( background_model );
  colour_mixture foreground( foreground_model );

  auto const colour_at = [&]( long y, long x, double( &out )[3] )
  {
    for( int p = 0; p < 3; ++p )
    {
      out[p] = image( static_cast< size_t >( x ), static_cast< size_t >( y ),
                      static_cast< size_t >( p ) );
    }
  };

  if( mode == grabcut_mode::WITH_RECT || mode == grabcut_mode::WITH_MASK )
  {
    // The mixtures start from k-means over each side's colours, seeded from a
    // generator in the state a fresh process has -- see the file comment on
    // why that is a decision and not a default.
    std::vector< float > background_samples, foreground_samples;
    background_samples.reserve( static_cast< size_t >( 3 * width * height ) );
    foreground_samples.reserve( static_cast< size_t >( 3 * width * height ) );

    for( long y = 0; y < height; ++y )
    {
      for( long x = 0; x < width; ++x )
      {
        auto& into = is_background(
          mask( static_cast< size_t >( x ), static_cast< size_t >( y ) ) )
          ? background_samples : foreground_samples;
        for( int p = 0; p < 3; ++p )
        {
          into.push_back( static_cast< float >(
            image( static_cast< size_t >( x ), static_cast< size_t >( y ),
                   static_cast< size_t >( p ) ) ) );
        }
      }
    }

    if( background_samples.empty() || foreground_samples.empty() )
    {
      throw std::invalid_argument(
        "grab_cut: the mask leaves one side with no pixels at all" );
    }

    mwc_rng rng( rng_state );
    auto const background_count =
      static_cast< int >( background_samples.size() / 3 );
    auto const foreground_count =
      static_cast< int >( foreground_samples.size() / 3 );
    auto const background_labels = kmeans_pp(
      background_samples, background_count,
      std::min( colour_mixture::components, background_count ), rng, 10 );
    auto const foreground_labels = kmeans_pp(
      foreground_samples, foreground_count,
      std::min( colour_mixture::components, foreground_count ), rng, 10 );

    background.begin_learning();
    for( int i = 0; i < background_count; ++i )
    {
      double const colour[3] = { background_samples[ 3 * i ],
                                 background_samples[ 3 * i + 1 ],
                                 background_samples[ 3 * i + 2 ] };
      background.add_sample( background_labels[ static_cast< size_t >( i ) ],
                             colour );
    }
    background.end_learning();

    foreground.begin_learning();
    for( int i = 0; i < foreground_count; ++i )
    {
      double const colour[3] = { foreground_samples[ 3 * i ],
                                 foreground_samples[ 3 * i + 1 ],
                                 foreground_samples[ 3 * i + 2 ] };
      foreground.add_sample( foreground_labels[ static_cast< size_t >( i ) ],
                             colour );
    }
    foreground.end_learning();
  }

  if( iterations <= 0 ) { return; }
  if( mode == grabcut_mode::EVAL_FREEZE_MODEL ) { iterations = 1; }

  constexpr double gamma = 50.0;
  constexpr double lambda = 9 * gamma;
  auto const beta = grabcut_beta( image );
  auto const weights = grabcut_neighbour_weights( image, beta, gamma );

  std::vector< int > component( static_cast< size_t >( width * height ), 0 );

  for( int pass = 0; pass < iterations; ++pass )
  {
    for( long y = 0; y < height; ++y )
    {
      for( long x = 0; x < width; ++x )
      {
        double colour[3];
        colour_at( y, x, colour );
        auto const label =
          mask( static_cast< size_t >( x ), static_cast< size_t >( y ) );
        component[ static_cast< size_t >( y * width + x ) ] =
          is_background( label ) ? background.which_component( colour )
                                 : foreground.which_component( colour );
      }
    }

    if( mode != grabcut_mode::EVAL_FREEZE_MODEL )
    {
      background.begin_learning();
      foreground.begin_learning();
      // Component by component, which is the order OpenCV adds them in and
      // therefore the order the double sums accumulate in.
      for( int c = 0; c < colour_mixture::components; ++c )
      {
        for( long y = 0; y < height; ++y )
        {
          for( long x = 0; x < width; ++x )
          {
            if( component[ static_cast< size_t >( y * width + x ) ] != c )
            { continue; }
            double colour[3];
            colour_at( y, x, colour );
            if( is_background(
                  mask( static_cast< size_t >( x ),
                        static_cast< size_t >( y ) ) ) )
            { background.add_sample( c, colour ); }
            else { foreground.add_sample( c, colour ); }
          }
        }
      }
      background.end_learning();
      foreground.end_learning();
    }

    flow_graph graph;
    graph.reserve(
      static_cast< size_t >( width * height ),
      static_cast< size_t >(
        2 * ( 4 * width * height - 3 * ( width + height ) + 2 ) ) );

    for( long y = 0; y < height; ++y )
    {
      for( long x = 0; x < width; ++x )
      {
        auto const vertex = graph.add_vertex();
        auto const i = static_cast< size_t >( y * width + x );
        auto const label =
          mask( static_cast< size_t >( x ), static_cast< size_t >( y ) );

        double from_source = 0.0;
        double to_sink = 0.0;
        if( is_probable( label ) )
        {
          double colour[3];
          colour_at( y, x, colour );
          from_source = -std::log( background.likelihood( colour ) );
          to_sink = -std::log( foreground.likelihood( colour ) );
        }
        else if( label == static_cast< uint8_t >( grabcut_label::BACKGROUND ) )
        {
          to_sink = lambda;
        }
        else { from_source = lambda; }

        graph.add_terminal_weights( vertex, from_source, to_sink );

        if( x > 0 )
        {
          graph.add_edges( vertex, vertex - 1, weights.left[i],
                           weights.left[i] );
        }
        if( x > 0 && y > 0 )
        {
          graph.add_edges( vertex, vertex - static_cast< int >( width ) - 1,
                           weights.up_left[i], weights.up_left[i] );
        }
        if( y > 0 )
        {
          graph.add_edges( vertex, vertex - static_cast< int >( width ),
                           weights.up[i], weights.up[i] );
        }
        if( x < width - 1 && y > 0 )
        {
          graph.add_edges( vertex, vertex - static_cast< int >( width ) + 1,
                           weights.up_right[i], weights.up_right[i] );
        }
      }
    }

    graph.max_flow();

    for( long y = 0; y < height; ++y )
    {
      for( long x = 0; x < width; ++x )
      {
        auto& label =
          mask( static_cast< size_t >( x ), static_cast< size_t >( y ) );
        if( !is_probable( label ) ) { continue; }
        label = graph.in_source_segment( static_cast< int >( y * width + x ) )
          ? static_cast< uint8_t >( grabcut_label::PROBABLY_FOREGROUND )
          : static_cast< uint8_t >( grabcut_label::PROBABLY_BACKGROUND );
      }
    }
  }
}

} // namespace image_kernels
} // namespace viame

#endif // VIAME_IMAGE_KERNELS_GRABCUT_H
