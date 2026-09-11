/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "disparity_segment.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace viame { namespace core {

bool fit_disparity_segment(
  const std::vector< std::pair< double, double > >& samples,
  int requested_samples, int max_outliers, double max_error,
  double& head_disparity, double& tail_disparity )
{
  if( requested_samples < 3 || requested_samples > 101 ||
      max_outliers < 0 || max_outliers >= ( requested_samples + 1 ) / 2 ||
      requested_samples - max_outliers < 3 ||
      !std::isfinite( max_error ) || max_error <= 0.0 ||
      samples.size() > static_cast< size_t >( requested_samples ) )
  {
    return false;
  }

  std::vector< std::pair< double, double > > valid;
  for( const auto& sample : samples )
  {
    if( std::isfinite( sample.first ) && std::isfinite( sample.second ) &&
        sample.first >= 0.0 && sample.first <= 1.0 && sample.second > 0.0 )
    {
      valid.push_back( sample );
    }
  }
  std::sort( valid.begin(), valid.end() );
  const size_t required = static_cast< size_t >( requested_samples - max_outliers );
  if( valid.size() < required ) { return false; }
  for( size_t i = 1; i < valid.size(); ++i )
  {
    if( valid[i].first - valid[i-1].first < 1e-9 ) { return false; }
  }

  // Deterministic two-point consensus; bounded to at most 101 samples.
  std::vector< size_t > best;
  double best_error = std::numeric_limits< double >::infinity();
  for( size_t i = 0; i < valid.size(); ++i )
  {
    for( size_t j = i + 1; j < valid.size(); ++j )
    {
      const double slope = ( valid[j].second - valid[i].second ) /
                           ( valid[j].first - valid[i].first );
      const double intercept = valid[i].second - slope * valid[i].first;
      std::vector< size_t > inliers;
      double error = 0.0;
      for( size_t k = 0; k < valid.size(); ++k )
      {
        const double residual = std::abs(
          valid[k].second - ( intercept + slope * valid[k].first ) );
        if( residual <= max_error )
        {
          inliers.push_back( k );
          error += residual * residual;
        }
      }
      if( inliers.size() < required ||
          valid[inliers.back()].first - valid[inliers.front()].first < 0.5 )
      {
        continue;
      }
      if( inliers.size() > best.size() ||
          ( inliers.size() == best.size() && error < best_error ) )
      {
        best = std::move( inliers );
        best_error = error;
      }
    }
  }
  if( best.empty() ) { return false; }

  // Least-squares fit of the consensus in disparity space. Reconstructing
  // endpoint correspondences lets the existing triangulator handle the rig.
  double mean_f = 0.0, mean_d = 0.0;
  for( auto i : best )
  {
    mean_f += valid[i].first;
    mean_d += valid[i].second;
  }
  mean_f /= best.size();
  mean_d /= best.size();
  double covariance = 0.0, variance = 0.0;
  for( auto i : best )
  {
    const double df = valid[i].first - mean_f;
    covariance += df * ( valid[i].second - mean_d );
    variance += df * df;
  }
  const double slope = covariance / variance;
  const double head = mean_d - slope * mean_f;
  const double tail = head + slope;
  if( !std::isfinite( head ) || !std::isfinite( tail ) || head <= 0.0 || tail <= 0.0 )
  {
    return false;
  }
  for( auto i : best )
  {
    if( std::abs( valid[i].second - ( head + slope * valid[i].first ) ) > max_error )
    {
      return false;
    }
  }
  head_disparity = head;
  tail_disparity = tail;
  return true;
}

} } // namespace viame::core
