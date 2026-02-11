/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief AdaBoost-specific IQR session subclass using OpenCV cv::ml::Boost.
 */

#ifndef VIAME_IQR_SESSION_ADABOOST_H
#define VIAME_IQR_SESSION_ADABOOST_H

#include <core/iqr_session.h>

#include <opencv2/ml.hpp>

#include <cstdio>
#include <fstream>

namespace viame
{

namespace iqr
{

class iqr_session_adaboost : public iqr_session
{
public:
  iqr_session_adaboost( unsigned pos_seed_neighbors )
    : iqr_session( pos_seed_neighbors )
  {}

  ~iqr_session_adaboost() override
  {
    free_model();
  }

  // -- AdaBoost-specific config setters --
  void set_boost_type( const std::string& type )
  {
    if( type == "discrete" ) m_boost_type = cv::ml::Boost::DISCRETE;
    else if( type == "real" ) m_boost_type = cv::ml::Boost::REAL;
    else if( type == "logit" ) m_boost_type = cv::ml::Boost::LOGIT;
    else if( type == "gentle" ) m_boost_type = cv::ml::Boost::GENTLE;
    else m_boost_type = cv::ml::Boost::GENTLE;
  }

  void set_weak_count( int count ) { m_weak_count = count; }
  void set_max_depth( int depth ) { m_max_depth = depth; }
  void set_weight_trim_rate( double rate ) { m_weight_trim_rate = rate; }

  // -- Virtual interface implementation --

  bool is_model_valid() const override
  {
    return m_boost_model != nullptr && !m_boost_model->empty();
  }

  void free_model() override
  {
    m_boost_model.release();
  }

  double predict_score( const std::vector< double >& vec ) const override
  {
    if( !is_model_valid() )
    {
      return compute_positive_similarity( vec );
    }

    // Convert to cv::Mat (1 x dim, CV_32F)
    cv::Mat sample( 1, static_cast< int >( vec.size() ), CV_32F );
    for( size_t i = 0; i < vec.size(); ++i )
    {
      sample.at< float >( 0, static_cast< int >( i ) ) =
        static_cast< float >( vec[i] );
    }

    // Predict with RAW_OUTPUT to get the weighted sum from weak classifiers
    float raw = m_boost_model->predict( sample, cv::noArray(),
      cv::ml::StatModel::RAW_OUTPUT );

    // Apply sigmoid to map raw output to [0, 1] probability
    // Positive class (label 1) gets high score when raw > 0
    double prob = 1.0 / ( 1.0 + std::exp( -static_cast< double >( raw ) ) );

    return prob;
  }

  double predict_distance( const std::vector< double >& vec ) const override
  {
    if( !is_model_valid() )
    {
      return 0.0;
    }

    // Convert to cv::Mat
    cv::Mat sample( 1, static_cast< int >( vec.size() ), CV_32F );
    for( size_t i = 0; i < vec.size(); ++i )
    {
      sample.at< float >( 0, static_cast< int >( i ) ) =
        static_cast< float >( vec[i] );
    }

    // Raw output: magnitude = confidence, sign = class
    float raw = m_boost_model->predict( sample, cv::noArray(),
      cv::ml::StatModel::RAW_OUTPUT );

    return static_cast< double >( raw );
  }

  std::vector< unsigned char > get_model_bytes() const override
  {
    if( !is_model_valid() )
    {
      return {};
    }

    // Save to temporary XML file
    const char* tmp_file = "tmp_adaboost_model.xml";
    m_boost_model->save( tmp_file );

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

    // Write to temporary XML file
    const char* tmp_file = "tmp_adaboost_model_load.xml";
    std::ofstream file( tmp_file, std::ios::binary );
    if( !file.is_open() )
    {
      return false;
    }
    file.write( reinterpret_cast< const char* >( bytes.data() ), bytes.size() );
    file.close();

    // Load model
    m_boost_model = cv::ml::Boost::load( tmp_file );
    std::remove( tmp_file );

    return is_model_valid();
  }

protected:
  std::string logger_name() const override
  {
    return "viame.opencv.process_query_adaboost";
  }

  bool train_model(
    const std::vector< descriptor_element >& auto_negatives ) override
  {
    size_t n_pos = m_positive_descriptors.size();
    size_t n_neg = m_negative_descriptors.size() + auto_negatives.size();
    size_t n_total = n_pos + n_neg;

    {
      auto logger = kwiver::vital::get_logger( logger_name() );
      LOG_INFO( logger, "AdaBoost training: " << n_pos << " positives, "
        << n_neg << " negatives (" << auto_negatives.size()
        << " auto-negatives)" );
    }

    if( n_total < 2 )
    {
      return false;
    }

    // Determine descriptor dimension from first positive
    size_t dim = m_positive_descriptors.begin()->second.vector.size();

    // Build training data as cv::Mat
    cv::Mat train_data( static_cast< int >( n_total ),
                        static_cast< int >( dim ), CV_32F );
    cv::Mat labels( static_cast< int >( n_total ), 1, CV_32S );

    int idx = 0;

    // Fill in positive samples (label 1)
    for( const auto& p : m_positive_descriptors )
    {
      for( size_t j = 0; j < dim; ++j )
      {
        train_data.at< float >( idx, static_cast< int >( j ) ) =
          static_cast< float >( p.second.vector[j] );
      }
      labels.at< int >( idx, 0 ) = 1;
      ++idx;
    }

    // Fill in negative samples (label 0)
    for( const auto& n : m_negative_descriptors )
    {
      for( size_t j = 0; j < dim; ++j )
      {
        train_data.at< float >( idx, static_cast< int >( j ) ) =
          static_cast< float >( n.second.vector[j] );
      }
      labels.at< int >( idx, 0 ) = 0;
      ++idx;
    }

    // Fill in auto-selected negative samples (label 0)
    for( const auto& n : auto_negatives )
    {
      for( size_t j = 0; j < dim; ++j )
      {
        train_data.at< float >( idx, static_cast< int >( j ) ) =
          static_cast< float >( n.vector[j] );
      }
      labels.at< int >( idx, 0 ) = 0;
      ++idx;
    }

    // Create training data wrapper
    cv::Ptr< cv::ml::TrainData > td = cv::ml::TrainData::create(
      train_data, cv::ml::ROW_SAMPLE, labels );

    // Configure AdaBoost
    m_boost_model = cv::ml::Boost::create();
    m_boost_model->setBoostType( m_boost_type );
    m_boost_model->setWeakCount( m_weak_count );
    m_boost_model->setMaxDepth( m_max_depth );
    m_boost_model->setWeightTrimRate( m_weight_trim_rate );

    // Train
    bool success = m_boost_model->train( td );

    if( !success || !is_model_valid() )
    {
      free_model();
      return false;
    }

    return true;
  }

private:
  cv::Ptr< cv::ml::Boost > m_boost_model;
  int m_boost_type = cv::ml::Boost::GENTLE;
  int m_weak_count = 100;
  int m_max_depth = 1;
  double m_weight_trim_rate = 0.95;
};

} // end namespace iqr
} // end namespace viame

#endif // VIAME_IQR_SESSION_ADABOOST_H
