/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Shared classification averaging implementation
 */

#include "utilities_target_clfr.h"

#include <map>
#include <string>
#include <vector>

namespace viame
{

namespace core
{

kv::detected_object_type_sptr
compute_average_classification(
  const std::vector< kv::detected_object_sptr >& detections,
  bool weighted,
  bool scale_by_conf,
  const std::string& ignore_class )
{
  std::map< std::string, double > class_sum;
  double ignore_sum = 0.0;

  double weighted_mass = 0.0;
  double weighted_non_ignore_mass = 0.0;
  double weighted_ignore_mass = 0.0;

  double conf_sum = 0.0;
  unsigned conf_count = 0;

  for( const auto& det : detections )
  {
    if( !det || !det->type() )
      continue;

    const auto& dot = det->type();
    double weight = weighted ? det->confidence() : 1.0;

    if( scale_by_conf )
    {
      conf_sum += det->confidence();
      conf_count += 1;
    }

    bool is_ignore = ( !ignore_class.empty() &&
                       dot->class_names().size() == 1 &&
                       dot->class_names()[0] == ignore_class );

    if( is_ignore )
    {
      ignore_sum += dot->score( ignore_class ) * weight;
      weighted_ignore_mass += weight;
    }
    else
    {
      for( const auto& name : dot->class_names() )
      {
        class_sum[name] += dot->score( name ) * weight;
      }
      weighted_non_ignore_mass += weight;
    }

    weighted_mass += weight;
  }

  // Confidence scaling factor: 0.1 + 0.9 * avg_confidence
  double prob_scale_factor = 1.0;

  if( scale_by_conf && conf_count > 0 )
  {
    prob_scale_factor = 0.1 + 0.9 * ( conf_sum / conf_count );
  }

  // Normalization depends on which classes are present
  if( weighted_mass > 0.0 && weighted_ignore_mass == 0.0 )
  {
    // No ignored detections — normalize by total mass
    prob_scale_factor /= weighted_mass;
  }
  else if( weighted_ignore_mass > 0.0 && weighted_non_ignore_mass > 0.0 )
  {
    // Mix of both — normalize by non-ignore mass, exclude ignore class
    prob_scale_factor /= weighted_non_ignore_mass;
  }
  else if( weighted_ignore_mass > 0.0 )
  {
    // Only ignore class — include it in output
    class_sum[ignore_class] = ignore_sum;
    prob_scale_factor /= weighted_ignore_mass;
  }

  if( class_sum.empty() )
    return kv::detected_object_type_sptr();

  std::vector< std::string > names;
  std::vector< double > scores;
  names.reserve( class_sum.size() );
  scores.reserve( class_sum.size() );

  for( const auto& entry : class_sum )
  {
    names.push_back( entry.first );
    scores.push_back( prob_scale_factor * entry.second );
  }

  return std::make_shared< kv::detected_object_type >( names, scores );
}

} // end namespace core

} // end namespace viame
