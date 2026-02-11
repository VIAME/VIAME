/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Abstract base class for IQR sessions with model-agnostic logic.
 *        Subclasses implement classifier-specific train/predict methods.
 */

#ifndef VIAME_IQR_SESSION_H
#define VIAME_IQR_SESSION_H

#include <core/utilities_iqr.h>

#include <vital/logger/logger.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_set>

namespace viame
{

namespace iqr
{

class iqr_session
{
public:
  iqr_session( unsigned pos_seed_neighbors )
    : m_pos_seed_neighbors( pos_seed_neighbors )
  {}

  virtual ~iqr_session() {}

  void reset()
  {
    m_positive_descriptors.clear();
    m_negative_descriptors.clear();
    m_working_index.clear();
    free_model();
  }

  // -- Shared config setters --
  void set_scoring_norm( bool val ) { m_scoring_norm = val; }
  void set_score_multiplier( double val ) { m_score_multiplier = val; }
  void set_nn_max_linear_search( unsigned max ) { m_nn_max_linear_search = max; }
  void set_nn_sample_fraction( double fraction ) { m_nn_sample_fraction = fraction; }
  void set_autoneg_select_ratio( unsigned ratio ) { m_autoneg_select_ratio = ratio; }
  void set_autoneg_from_full_index( bool from_full ) { m_autoneg_from_full_index = from_full; }
  void set_full_index_ref(
    const std::unordered_map< std::string, std::vector< double > >* index )
  {
    m_full_index_ref = index;
  }
  void set_lsh_index_ref( const lsh_index* index ) { m_lsh_index_ref = index; }
  void set_lsh_neighbor_multiplier( unsigned mult ) { m_lsh_neighbor_multiplier = mult; }
  void set_nn_distance_method( const std::string& method ) { m_nn_distance_method = method; }
  void set_force_exemplar_scores( bool force ) { m_force_exemplar_scores = force; }

  size_t num_positives() const { return m_positive_descriptors.size(); }
  size_t num_negatives() const { return m_negative_descriptors.size(); }

  // -- Pure virtual interface for subclasses --
  virtual bool is_model_valid() const = 0;
  virtual void free_model() = 0;
  virtual double predict_score( const std::vector< double >& vec ) const = 0;
  virtual double predict_distance( const std::vector< double >& vec ) const = 0;
  virtual std::vector< unsigned char > get_model_bytes() const = 0;
  virtual bool load_model_from_bytes( const std::vector< unsigned char >& bytes ) = 0;

  // -- Concrete model-agnostic methods --

  void adjudicate( const std::vector< descriptor_element >& positives,
                   const std::vector< descriptor_element >& negatives = {} )
  {
    for( const auto& desc : positives )
    {
      m_positive_descriptors[desc.uid] = desc;
    }
    for( const auto& desc : negatives )
    {
      m_negative_descriptors[desc.uid] = desc;
    }
  }

  void update_working_index(
    const std::unordered_map< std::string, std::vector< double > >& full_index )
  {
    // Add all positive and negative descriptors to working index
    for( const auto& p : m_positive_descriptors )
    {
      m_working_index[p.first] = p.second;
    }
    for( const auto& n : m_negative_descriptors )
    {
      m_working_index[n.first] = n.second;
    }

    // For each positive, find nearest neighbors and add to working index
    for( const auto& pos : m_positive_descriptors )
    {
      auto neighbors = find_nearest_neighbors(
        pos.second.vector, full_index, m_pos_seed_neighbors );

      for( const auto& neighbor : neighbors )
      {
        if( m_working_index.find( neighbor.first ) == m_working_index.end() )
        {
          m_working_index[neighbor.first] = descriptor_element(
            neighbor.first, full_index.at( neighbor.first ) );
        }
      }
    }

    // For each negative, find nearest neighbors and add to working index
    for( const auto& neg : m_negative_descriptors )
    {
      auto neighbors = find_nearest_neighbors(
        neg.second.vector, full_index, m_pos_seed_neighbors );

      for( const auto& neighbor : neighbors )
      {
        if( m_working_index.find( neighbor.first ) == m_working_index.end() )
        {
          m_working_index[neighbor.first] = descriptor_element(
            neighbor.first, full_index.at( neighbor.first ) );
        }
      }
    }

    auto logger = kwiver::vital::get_logger( logger_name() );
    LOG_INFO( logger, "Working index size after neighbor expansion: "
      << m_working_index.size()
      << " (positives: " << m_positive_descriptors.size()
      << ", negatives: " << m_negative_descriptors.size() << ")" );
  }

  bool refine()
  {
    if( m_positive_descriptors.empty() )
    {
      return false;
    }

    // Free existing model
    free_model();

    // Auto-select negatives if none provided
    std::vector< descriptor_element > auto_negatives;
    if( m_negative_descriptors.empty() && m_autoneg_select_ratio > 0 )
    {
      auto_negatives = select_auto_negatives();
    }

    // If still no negatives (manual or auto), return true but no model
    if( m_negative_descriptors.empty() && auto_negatives.empty() )
    {
      auto logger = kwiver::vital::get_logger( logger_name() );
      LOG_WARN( logger, "No negatives available and autoneg_select_ratio="
        << m_autoneg_select_ratio << ", skipping model training" );
      return true;
    }

    return train_model( auto_negatives );
  }

  // Get ordered results from working index
  std::vector< std::pair< std::string, double > > ordered_results()
  {
    std::vector< std::pair< std::string, double > > results;

    // First, score ALL items in working index
    for( const auto& entry : m_working_index )
    {
      double score = predict_score( entry.second.vector );
      results.emplace_back( entry.first, score );
    }

    if( !results.empty() )
    {
      auto logger = kwiver::vital::get_logger( logger_name() );

      bool used_model = is_model_valid();
      LOG_INFO( logger, "Scoring method: "
        << ( used_model ? "trained model" : "fallback similarity" ) );

      // Compute score statistics
      double min_score = results[0].second;
      double max_score = results[0].second;
      for( const auto& r : results )
      {
        if( r.second < min_score ) min_score = r.second;
        if( r.second > max_score ) max_score = r.second;
      }

      // Compute median
      std::vector< double > scores;
      scores.reserve( results.size() );
      for( const auto& r : results )
      {
        scores.push_back( r.second );
      }
      std::sort( scores.begin(), scores.end() );
      double median_score = scores[scores.size() / 2];

      LOG_INFO( logger, "Score distribution (" << results.size()
        << " items): min=" << min_score << ", max=" << max_score
        << ", median=" << median_score );
    }

    // Check if probabilities need to be inverted
    // If positive exemplars have lower average probability than overall average,
    // the model labels may be flipped
    if( is_model_valid() && !m_positive_descriptors.empty() && !results.empty() )
    {
      double pos_prob_sum = 0.0;
      for( const auto& p : m_positive_descriptors )
      {
        pos_prob_sum += predict_score( p.second.vector );
      }
      double pos_avg = pos_prob_sum / m_positive_descriptors.size();

      double all_prob_sum = 0.0;
      for( const auto& r : results )
      {
        all_prob_sum += r.second;
      }
      double all_avg = all_prob_sum / results.size();

      // If positive examples have lower average probability, invert all scores
      if( pos_avg < all_avg )
      {
        auto logger = kwiver::vital::get_logger( logger_name() );
        LOG_INFO( logger, "Score inversion triggered (pos_avg="
          << pos_avg << " < all_avg=" << all_avg << ")" );
        for( auto& r : results )
        {
          r.second = 1.0 - r.second;
        }
        m_invert_probabilities = true;
      }
      else
      {
        m_invert_probabilities = false;
      }
    }

    // Optionally force positive exemplars to score 1.0 and negative to 0.0
    // This is done AFTER the inversion check
    if( m_force_exemplar_scores )
    {
      for( auto& r : results )
      {
        if( m_positive_descriptors.find( r.first ) != m_positive_descriptors.end() )
        {
          r.second = 1.0;
        }
        else if( m_negative_descriptors.find( r.first ) != m_negative_descriptors.end() )
        {
          r.second = 0.0;
        }
      }
    }

    // Sort by score descending
    std::sort( results.begin(), results.end(),
      []( const auto& a, const auto& b ) { return a.second > b.second; } );

    return results;
  }

  // Get feedback descriptors (closest to decision boundary, or by similarity if no model)
  std::vector< std::pair< std::string, double > > ordered_feedback() const
  {
    bool has_valid_model = is_model_valid();

    std::vector< std::pair< std::string, double > > feedback;

    for( const auto& entry : m_working_index )
    {
      if( has_valid_model )
      {
        // With model: use distance to decision boundary
        double distance = predict_distance( entry.second.vector );
        feedback.emplace_back( entry.first, std::abs( distance ) );
      }
      else
      {
        // Without model: use similarity to positive exemplars
        // Lower similarity = more uncertain = better feedback candidate
        double similarity = compute_positive_similarity( entry.second.vector );
        feedback.emplace_back( entry.first, similarity );
      }
    }

    if( has_valid_model )
    {
      // Sort by absolute distance ascending (closest to boundary first)
      std::sort( feedback.begin(), feedback.end(),
        []( const auto& a, const auto& b ) { return a.second < b.second; } );
    }
    else
    {
      // Sort by similarity - medium similarity samples are best for feedback
      // (not too similar to positives, not too dissimilar)
      // Sort descending so highest similarity first, then take from middle
      std::sort( feedback.begin(), feedback.end(),
        []( const auto& a, const auto& b ) { return a.second > b.second; } );

      // Reorder to prioritize samples near the median similarity
      if( feedback.size() > 2 )
      {
        size_t mid = feedback.size() / 2;
        std::vector< std::pair< std::string, double > > reordered;
        reordered.reserve( feedback.size() );

        // Interleave from middle outward
        size_t left = mid;
        size_t right = mid + 1;
        bool pick_left = true;

        while( reordered.size() < feedback.size() )
        {
          if( pick_left && left > 0 )
          {
            --left;
            reordered.push_back( feedback[left] );
          }
          else if( !pick_left && right < feedback.size() )
          {
            reordered.push_back( feedback[right] );
            ++right;
          }
          else if( left > 0 )
          {
            --left;
            reordered.push_back( feedback[left] );
          }
          else if( right < feedback.size() )
          {
            reordered.push_back( feedback[right] );
            ++right;
          }
          else
          {
            // Safety break to prevent infinite loop
            break;
          }
          pick_left = !pick_left;
        }
        feedback = std::move( reordered );
      }
    }

    return feedback;
  }

  double compute_positive_similarity( const std::vector< double >& vec ) const
  {
    if( m_positive_descriptors.empty() )
    {
      return 0.0;
    }

    // Compute mean similarity to positive descriptors
    double total_similarity = 0.0;
    for( const auto& p : m_positive_descriptors )
    {
      double dist = compute_distance( vec, p.second.vector );

      // Convert distance to similarity
      double sim = 0.0;
      if( m_nn_distance_method == "euclidean" )
      {
        // Convert Euclidean distance to similarity in [0, 1]
        // arithmetic_mean ( 1 / (1 + d ) )
        sim = 1.0 / ( 1.0 + ( dist * m_score_multiplier ) );
      }
      else if( m_nn_distance_method == "cosine" )
      {
        // Cosine distance is in [0, 1] (for positive vectors), so similarity = 1 - distance
        sim = 1.0 - dist;
      }
      else
      {
        // HIK or other: distance is already "inverted" similarity?
        // HIK distance = 1.0 - intersection.
        // Intersection is similarity. So sim = 1.0 - dist
        sim = 1.0 - dist;
      }

      total_similarity += sim;
    }

    return total_similarity / m_positive_descriptors.size();
  }

protected:
  // Returns the logger name for this session type
  virtual std::string logger_name() const = 0;

  // Subclass implements the actual model training.
  // Receives auto-negatives (may be empty if manual negatives exist).
  // Can access m_positive_descriptors and m_negative_descriptors directly.
  virtual bool train_model(
    const std::vector< descriptor_element >& auto_negatives ) = 0;

  // -- Auto-negative selection --

  std::vector< descriptor_element > select_auto_negatives() const
  {
    if( m_positive_descriptors.empty() || m_autoneg_select_ratio == 0 )
    {
      return {};
    }

    // Verify descriptor norm
    auto logger = kwiver::vital::get_logger( logger_name() );
    for( const auto& p : m_positive_descriptors )
    {
      double norm_sq = 0.0;
      for( double v : p.second.vector )
      {
        norm_sq += v * v;
      }
      double norm = std::sqrt( norm_sq );
      LOG_INFO( logger, "Positive Exemplar (Query) Norm: " << norm );
      LOG_INFO( logger, "Exemplar UID: " << p.first );
    }

    // Choose which index to select auto-negatives from
    bool use_full_index = m_autoneg_from_full_index && m_full_index_ref &&
                          !m_full_index_ref->empty();

    if( !use_full_index && m_working_index.empty() )
    {
      return {};
    }

    std::unordered_set< std::string > selected_uids;
    std::vector< descriptor_element > auto_negatives;

    // For each positive, find the most distant descriptors
    for( const auto& pos : m_positive_descriptors )
    {
      // Priority queue to track most distant descriptors (min-heap by distance)
      // We use min-heap so we can efficiently maintain top-k most distant
      using pair_type = std::pair< double, std::string >;
      std::priority_queue< pair_type, std::vector< pair_type >,
        std::greater< pair_type > > pq;

      if( use_full_index )
      {
        // Select from full descriptor index
        for( const auto& entry : *m_full_index_ref )
        {
          // Skip if already a positive or negative or already selected
          if( m_positive_descriptors.find( entry.first ) !=
              m_positive_descriptors.end() )
          {
            continue;
          }
          if( m_negative_descriptors.find( entry.first ) !=
              m_negative_descriptors.end() )
          {
            continue;
          }
          if( selected_uids.find( entry.first ) != selected_uids.end() )
          {
            continue;
          }

          // Use histogram intersection distance
          double dist = histogram_intersection_distance(
            pos.second.vector, entry.second );

          if( pq.size() < m_autoneg_select_ratio )
          {
            pq.push( { dist, entry.first } );
          }
          else if( dist > pq.top().first )
          {
            pq.pop();
            pq.push( { dist, entry.first } );
          }
        }

        // Add most distant descriptors as auto-negatives
        while( !pq.empty() )
        {
          const auto& uid = pq.top().second;
          if( selected_uids.find( uid ) == selected_uids.end() )
          {
            selected_uids.insert( uid );
            auto_negatives.emplace_back(
              descriptor_element( uid, m_full_index_ref->at( uid ) ) );
          }
          pq.pop();
        }
      }
      else
      {
        // Select from working index (default, matches SMQTK behavior)
        for( const auto& entry : m_working_index )
        {
          // Skip if already a positive or negative or already selected
          if( m_positive_descriptors.find( entry.first ) !=
              m_positive_descriptors.end() )
          {
            continue;
          }
          if( m_negative_descriptors.find( entry.first ) !=
              m_negative_descriptors.end() )
          {
            continue;
          }
          if( selected_uids.find( entry.first ) != selected_uids.end() )
          {
            continue;
          }

          // Use histogram intersection distance
          double dist = histogram_intersection_distance(
            pos.second.vector, entry.second.vector );

          if( pq.size() < m_autoneg_select_ratio )
          {
            pq.push( { dist, entry.first } );
          }
          else if( dist > pq.top().first )
          {
            pq.pop();
            pq.push( { dist, entry.first } );
          }
        }

        // Add most distant descriptors as auto-negatives
        while( !pq.empty() )
        {
          const auto& uid = pq.top().second;
          if( selected_uids.find( uid ) == selected_uids.end() )
          {
            selected_uids.insert( uid );
            auto_negatives.emplace_back( m_working_index.at( uid ) );
          }
          pq.pop();
        }
      }
    }

    return auto_negatives;
  }

  // -- Distance computation --

  double compute_distance( const std::vector< double >& a,
                           const std::vector< double >& b ) const
  {
    if( m_nn_distance_method == "euclidean" && m_scoring_norm )
    {
      // Enforce normalization to match Baseline/Cosine behavior
      // This is required because input descriptors are unnormalized (ReLU outputs)
      // and Euclidean search on unnormalized data yields poor ranking.
      std::vector< double > a_norm( a.size() );
      double norm_a_sq = 0.0;
      for( double val : a ) norm_a_sq += val * val;
      double norm_a = ( norm_a_sq > 0 ) ? std::sqrt( norm_a_sq ) : 1.0;
      for( size_t i = 0; i < a.size(); ++i ) a_norm[i] = a[i] / norm_a;

      std::vector< double > b_norm( b.size() );
      double norm_b_sq = 0.0;
      for( double val : b ) norm_b_sq += val * val;
      double norm_b = ( norm_b_sq > 0 ) ? std::sqrt( norm_b_sq ) : 1.0;
      for( size_t i = 0; i < b.size(); ++i ) b_norm[i] = b[i] / norm_b;

      return euclidean_distance( a_norm, b_norm );
    }
    else if( m_nn_distance_method == "cosine" )
    {
      return cosine_distance( a, b );
    }
    else // "hik" or default
    {
      return histogram_intersection_distance( a, b );
    }
  }

  // -- Nearest neighbor search --

  std::vector< std::pair< std::string, double > > find_nearest_neighbors(
    const std::vector< double >& query,
    const std::unordered_map< std::string, std::vector< double > >& index,
    size_t k ) const
  {
    // Priority queue for k nearest neighbors (max heap by distance)
    using pair_type = std::pair< double, std::string >;
    std::priority_queue< pair_type > pq;

    // Try LSH-based search first for fast approximate nearest neighbors
    // Algorithm: get k*multiplier unique hashes, expand to all UIDs, re-rank by distance
    if( m_lsh_index_ref && m_lsh_index_ref->is_loaded() )
    {
      // Get k*multiplier unique hashes, expand to all UIDs with those hashes
      // The multiplier increases the candidate pool for better recall
      size_t num_hashes = k * m_lsh_neighbor_multiplier;
      auto expanded_uids = m_lsh_index_ref->find_neighbors_by_hash( query, num_hashes );

      // Re-rank candidates using configured distance method
      for( const std::string& uid : expanded_uids )
      {
        auto it = index.find( uid );
        if( it == index.end() )
        {
          continue;
        }

        double dist = compute_distance( query, it->second );

        if( pq.size() < k )
        {
          pq.push( { dist, uid } );
        }
        else if( dist < pq.top().first )
        {
          pq.pop();
          pq.push( { dist, uid } );
        }
      }
    }
    else
    {
      // Fall back to brute-force or random sampling
      bool use_approximate = ( index.size() > m_nn_max_linear_search );

      if( use_approximate )
      {
        // Approximate search: sample a fraction of the index
        size_t sample_size = static_cast< size_t >(
          index.size() * m_nn_sample_fraction );
        sample_size = std::max( sample_size, k * 2 ); // At least 2x k
        sample_size = std::min( sample_size, index.size() ); // Cap at index size

        // Create a vector of keys and sample randomly
        std::vector< std::string > keys;
        keys.reserve( index.size() );
        for( const auto& entry : index )
        {
          keys.push_back( entry.first );
        }

        // Shuffle and take first sample_size keys
        std::random_device rd;
        std::mt19937 gen( rd() );
        std::shuffle( keys.begin(), keys.end(), gen );

        for( size_t i = 0; i < sample_size; ++i )
        {
          const auto& key = keys[i];
          const auto& vec = index.at( key );
          double dist = compute_distance( query, vec );

          if( pq.size() < k )
          {
            pq.push( { dist, key } );
          }
          else if( dist < pq.top().first )
          {
            pq.pop();
            pq.push( { dist, key } );
          }
        }
      }
      else
      {
        // Exact linear search
        for( const auto& entry : index )
        {
          double dist = compute_distance( query, entry.second );

          if( pq.size() < k )
          {
            pq.push( { dist, entry.first } );
          }
          else if( dist < pq.top().first )
          {
            pq.pop();
            pq.push( { dist, entry.first } );
          }
        }
      }
    }

    // Convert to vector
    std::vector< std::pair< std::string, double > > result;
    result.reserve( pq.size() );
    while( !pq.empty() )
    {
      result.emplace_back( pq.top().second, pq.top().first );
      pq.pop();
    }

    // Reverse to get closest first
    std::reverse( result.begin(), result.end() );
    return result;
  }

  // -- Protected members --
  unsigned m_pos_seed_neighbors;
  unsigned m_nn_max_linear_search = 50000;
  double m_nn_sample_fraction = 0.1;
  unsigned m_autoneg_select_ratio = 0;
  bool m_autoneg_from_full_index = false;
  std::string m_nn_distance_method = "euclidean";
  bool m_force_exemplar_scores = true;
  bool m_scoring_norm = true;
  double m_score_multiplier = 1.0;

  std::unordered_map< std::string, descriptor_element > m_positive_descriptors;
  std::unordered_map< std::string, descriptor_element > m_negative_descriptors;
  std::unordered_map< std::string, descriptor_element > m_working_index;

  // Reference to full index for auto-negative selection
  const std::unordered_map< std::string, std::vector< double > >* m_full_index_ref = nullptr;

  // Reference to LSH index for fast approximate NN search
  const lsh_index* m_lsh_index_ref = nullptr;
  unsigned m_lsh_neighbor_multiplier = 10;

  // Whether to invert probabilities (determined during ordered_results)
  bool m_invert_probabilities = false;
};

} // end namespace iqr
} // end namespace viame

#endif // VIAME_IQR_SESSION_H
