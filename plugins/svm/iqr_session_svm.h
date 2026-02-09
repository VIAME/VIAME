/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief SVM-specific IQR session subclass using libsvm.
 */

#ifndef VIAME_IQR_SESSION_SVM_H
#define VIAME_IQR_SESSION_SVM_H

#include <core/iqr_session.h>

#include <svm.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

namespace viame
{

namespace iqr
{

class iqr_session_svm : public iqr_session
{
public:
  iqr_session_svm( unsigned pos_seed_neighbors )
    : iqr_session( pos_seed_neighbors )
    , m_svm_model( nullptr )
  {
    // Suppress libSVM output
    svm_set_print_string_function( []( const char* ) {} );
  }

  ~iqr_session_svm() override
  {
    free_model();
  }

  // -- SVM-specific config setters --
  void set_kernel_type( const std::string& type )
  {
    if( type == "linear" ) m_kernel_type = LINEAR;
    else if( type == "poly" ) m_kernel_type = POLY;
    else if( type == "rbf" ) m_kernel_type = RBF;
    else if( type == "sigmoid" ) m_kernel_type = SIGMOID;
    else if( type == "histogram" ) m_kernel_type = HISTOGRAM;
    else m_kernel_type = HISTOGRAM;
  }

  void set_c( double c ) { m_c = c; }
  void set_gamma( double gamma ) { m_gamma = gamma; }
  void set_use_platt_scaling( bool use_platt ) { m_use_platt_scaling = use_platt; }

  // -- Virtual interface implementation --

  bool is_model_valid() const override
  {
    if( !m_svm_model )
    {
      return false;
    }

    if( m_svm_model->l <= 0 ||
        m_svm_model->SV == nullptr ||
        m_svm_model->sv_coef == nullptr ||
        m_svm_model->rho == nullptr ||
        m_svm_model->nSV == nullptr ||
        m_svm_model->nr_class < 2 )
    {
      return false;
    }

    // Check sv_coef sub-arrays
    for( int i = 0; i < m_svm_model->nr_class - 1; ++i )
    {
      if( m_svm_model->sv_coef[i] == nullptr )
      {
        return false;
      }
    }

    // probA and probB are required for svm_predict_probability to fill
    // prob_estimates. Without them it silently returns the predicted label
    // and leaves prob_estimates uninitialized.
    if( m_svm_model->probA == nullptr || m_svm_model->probB == nullptr )
    {
      return false;
    }

    return true;
  }

  void free_model() override
  {
    if( m_svm_model )
    {
      svm_free_and_destroy_model( &m_svm_model );
      m_svm_model = nullptr;
    }
  }

  double predict_score( const std::vector< double >& vec ) const override
  {
    if( m_use_platt_scaling )
    {
      return predict_score_platt( vec );
    }
    else
    {
      return predict_score_libsvm( vec );
    }
  }

  double predict_distance( const std::vector< double >& vec ) const override
  {
    if( !is_model_valid() )
    {
      return 0.0;
    }

    svm_node* nodes = allocate_svm_nodes( vec );

    double dec_values[1];
    svm_predict_values( m_svm_model, nodes, dec_values );

    delete[] nodes;
    return dec_values[0];
  }

  std::vector< unsigned char > get_model_bytes() const override
  {
    if( !is_model_valid() )
    {
      return {};
    }

    // Save to temporary file
    const char* tmp_file = "tmp_svm_model.bin";
    if( svm_save_model( tmp_file, m_svm_model ) != 0 )
    {
      return {};
    }

    // Read file contents
    std::ifstream file( tmp_file, std::ios::binary | std::ios::ate );
    if( !file.is_open() )
    {
      std::remove( tmp_file );
      return {};
    }

    std::streamsize size = file.tellg();
    file.seekg( 0, std::ios::beg );

    std::vector< unsigned char > bytes( size );
    file.read( reinterpret_cast< char* >( bytes.data() ), size );
    file.close();

    std::remove( tmp_file );
    return bytes;
  }

  bool load_model_from_bytes( const std::vector< unsigned char >& bytes ) override
  {
    if( bytes.empty() )
    {
      return false;
    }

    free_model();

    // Write to temporary file
    const char* tmp_file = "tmp_svm_model_load.bin";
    std::ofstream file( tmp_file, std::ios::binary );
    if( !file.is_open() )
    {
      return false;
    }
    file.write( reinterpret_cast< const char* >( bytes.data() ), bytes.size() );
    file.close();

    // Load model
    m_svm_model = svm_load_model( tmp_file );
    std::remove( tmp_file );

    return m_svm_model != nullptr;
  }

protected:
  std::string logger_name() const override
  {
    return "viame.svm.process_query";
  }

  bool train_model(
    const std::vector< descriptor_element >& auto_negatives ) override
  {
    // Build SVM training data
    size_t n_pos = m_positive_descriptors.size();
    size_t n_neg = m_negative_descriptors.size() + auto_negatives.size();
    size_t n_total = n_pos + n_neg;

    {
      auto logger = kwiver::vital::get_logger( logger_name() );
      LOG_INFO( logger, "SVM training: " << n_pos << " positives, "
        << n_neg << " negatives (" << auto_negatives.size()
        << " auto-negatives)" );
    }

    if( n_total < 2 )
    {
      return false;
    }

    // Allocate SVM problem
    svm_problem prob;
    prob.l = static_cast< int >( n_total );
    prob.y = new double[n_total];
    prob.x = new svm_node*[n_total];

    // Fill in positive samples (label +1)
    size_t idx = 0;
    for( const auto& p : m_positive_descriptors )
    {
      prob.y[idx] = 1.0;
      prob.x[idx] = allocate_svm_nodes( p.second.vector );
      ++idx;
    }

    // Fill in negative samples (label -1)
    for( const auto& n : m_negative_descriptors )
    {
      prob.y[idx] = -1.0;
      prob.x[idx] = allocate_svm_nodes( n.second.vector );
      ++idx;
    }

    // Fill in auto-selected negative samples (label -1)
    for( const auto& n : auto_negatives )
    {
      prob.y[idx] = -1.0;
      prob.x[idx] = allocate_svm_nodes( n.vector );
      ++idx;
    }

    // Set SVM parameters
    svm_parameter param;
    std::memset( &param, 0, sizeof( param ) );
    param.svm_type = C_SVC;
    param.kernel_type = m_kernel_type;
    param.degree = 3;
    param.gamma = m_gamma;
    param.coef0 = 0;
    param.cache_size = 200;
    param.eps = 0.001;
    param.C = m_c;
    param.nu = 0.5;
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = 1;

    // Class weighting to handle imbalanced pos/neg samples
    // Weight for positive class = max(1.0, num_neg / num_pos)
    int weight_labels[2] = { 1, -1 };
    double weights[2] = {
      std::max( 1.0, static_cast< double >( n_neg ) / n_pos ),
      1.0
    };
    param.nr_weight = 2;
    param.weight_label = weight_labels;
    param.weight = weights;

    // Check parameters
    const char* error = svm_check_parameter( &prob, &param );
    if( error )
    {
      // Clean up and return
      for( size_t i = 0; i < n_total; ++i )
      {
        delete[] prob.x[i];
      }
      delete[] prob.x;
      delete[] prob.y;
      return false;
    }

    // Train model
    // Set random seed for deterministic libsvm behavior (libsvm uses rand() for shuffling)
    std::srand( 0 );
    m_svm_model = svm_train( &prob, &param );

    // Clean up training data
    for( size_t i = 0; i < n_total; ++i )
    {
      delete[] prob.x[i];
    }
    delete[] prob.x;
    delete[] prob.y;

    // Validate model is usable (has all required fields)
    if( m_svm_model && !is_model_valid() )
    {
      free_model();
    }

    return m_svm_model != nullptr;
  }

private:
  svm_node* allocate_svm_nodes( const std::vector< double >& vec ) const
  {
    svm_node* nodes = new svm_node[vec.size() + 1];
    for( size_t i = 0; i < vec.size(); ++i )
    {
      nodes[i].index = static_cast< int >( i + 1 );
      nodes[i].value = vec[i];
    }
    nodes[vec.size()].index = -1;
    nodes[vec.size()].value = 0;
    return nodes;
  }

  // Custom Platt scaling using histogram intersection kernel.
  // Computes: f(x) = sum(sv_coef[i] * K(SV[i], x)) - rho
  // where K = sum(min(a[i], b[i])) is the HIK kernel value
  // Then applies Platt scaling: prob = 1 / (1 + exp(A * f + B))
  double predict_score_platt( const std::vector< double >& vec ) const
  {
    if( !is_model_valid() )
    {
      return compute_positive_similarity( vec );
    }

    // Safety checks
    if( !m_svm_model->SV || !m_svm_model->sv_coef || !m_svm_model->rho )
    {
      return compute_positive_similarity( vec );
    }

    // Get model parameters
    int num_svs = m_svm_model->l;
    if( num_svs <= 0 )
    {
      return compute_positive_similarity( vec );
    }

    double rho = m_svm_model->rho[0];
    double probA = m_svm_model->probA ? m_svm_model->probA[0] : 0.0;
    double probB = m_svm_model->probB ? m_svm_model->probB[0] : 0.0;

    // Compute decision value: f(x) = sum(sv_coef[i] * K(SV[i], x)) - rho
    // where K is the histogram intersection kernel = sum(min(a[i], b[i]))
    double decision_value = 0.0;
    size_t vec_dim = vec.size();

    for( int i = 0; i < num_svs; ++i )
    {
      if( !m_svm_model->SV[i] || !m_svm_model->sv_coef[0] )
      {
        continue;
      }

      // Pre-allocate support vector with zeros to match input dimension
      std::vector< double > sv( vec_dim, 0.0 );

      // Extract non-zero elements from sparse SV representation
      for( int j = 0; m_svm_model->SV[i][j].index != -1; ++j )
      {
        int idx = m_svm_model->SV[i][j].index - 1;  // libsvm uses 1-based indexing
        if( idx >= 0 && idx < static_cast< int >( vec_dim ) )
        {
          sv[idx] = m_svm_model->SV[i][j].value;
        }
      }

      // Compute histogram intersection KERNEL value (NOT distance)
      // HIK kernel = sum(min(a[i], b[i]))
      // histogram_intersection_distance returns 1.0 - kernel_value
      double kernel_value = 1.0 - histogram_intersection_distance( sv, vec );

      // sv_coef is alpha * y for each SV (for binary classification, sv_coef[0])
      decision_value += m_svm_model->sv_coef[0][i] * kernel_value;
    }

    decision_value -= rho;

    // Apply Platt scaling: prob = 1 / (1 + exp(A * f + B))
    // where f is the decision value, A = probA, B = probB
    double prob = 1.0 / ( 1.0 + std::exp( probA * decision_value + probB ) );

    return prob;
  }

  // Use libsvm's built-in probability prediction
  double predict_score_libsvm( const std::vector< double >& vec ) const
  {
    if( !is_model_valid() )
    {
      return compute_positive_similarity( vec );
    }

    svm_node* nodes = allocate_svm_nodes( vec );

    double prob_estimates[2];
    svm_predict_probability( m_svm_model, nodes, prob_estimates );

    delete[] nodes;

    // Get label order
    int labels[2];
    svm_get_labels( m_svm_model, labels );

    // Return probability for positive class (label +1)
    return ( labels[0] == 1 ) ? prob_estimates[0] : prob_estimates[1];
  }

  int m_kernel_type = HISTOGRAM;
  double m_c = 2.0;
  double m_gamma = 0.0078125;
  bool m_use_platt_scaling = false;

  svm_model* m_svm_model;
};

} // end namespace iqr
} // end namespace viame

#endif // VIAME_IQR_SESSION_SVM_H
