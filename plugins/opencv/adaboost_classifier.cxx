/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "adaboost_classifier.h"

#include <opencv2/core/persistence.hpp>
#include <fstream>
#include <sstream>

namespace viame {
namespace opencv {

//------------------------------------------------------------------------------
// AdaBoostClassifier Implementation
//------------------------------------------------------------------------------

AdaBoostClassifier::AdaBoostClassifier()
  : m_config()
  , m_featureDimension( 0 )
  , m_trained( false )
{
}

AdaBoostClassifier::AdaBoostClassifier( const AdaBoostConfig& config )
  : m_config( config )
  , m_featureDimension( 0 )
  , m_trained( false )
{
}

AdaBoostClassifier::~AdaBoostClassifier()
{
}

void AdaBoostClassifier::setConfig( const AdaBoostConfig& config )
{
  m_config = config;
}

AdaBoostConfig AdaBoostClassifier::getConfig() const
{
  return m_config;
}

int AdaBoostClassifier::getOpenCVBoostType() const
{
  switch( m_config.boostType )
  {
    case AdaBoostType::DISCRETE:
      return cv::ml::Boost::DISCRETE;
    case AdaBoostType::REAL:
      return cv::ml::Boost::REAL;
    case AdaBoostType::LOGIT:
      return cv::ml::Boost::LOGIT;
    case AdaBoostType::GENTLE:
    default:
      return cv::ml::Boost::GENTLE;
  }
}

cv::Ptr< cv::ml::Boost > AdaBoostClassifier::createBoostClassifier() const
{
  cv::Ptr< cv::ml::Boost > boost = cv::ml::Boost::create();

  boost->setBoostType( getOpenCVBoostType() );
  boost->setWeakCount( m_config.weakCount );
  boost->setWeightTrimRate( m_config.weightTrimRate );
  boost->setMaxDepth( m_config.maxDepth );
  boost->setUseSurrogates( m_config.useSurrogates );
  boost->setMinSampleCount( m_config.minSampleCount );

  return boost;
}

bool AdaBoostClassifier::train( const cv::Mat& features, const cv::Mat& labels )
{
  if( features.empty() || labels.empty() )
  {
    return false;
  }

  if( features.rows != labels.rows )
  {
    return false;
  }

  // Convert features to float if needed
  cv::Mat featuresFloat;
  if( features.type() != CV_32F )
  {
    features.convertTo( featuresFloat, CV_32F );
  }
  else
  {
    featuresFloat = features;
  }

  // Convert labels to int if needed
  cv::Mat labelsInt;
  if( labels.type() != CV_32S )
  {
    labels.convertTo( labelsInt, CV_32S );
  }
  else
  {
    labelsInt = labels;
  }

  // Create training data
  cv::Ptr< cv::ml::TrainData > trainData = cv::ml::TrainData::create(
    featuresFloat, cv::ml::ROW_SAMPLE, labelsInt );

  // Create and train classifier
  m_classifier = createBoostClassifier();

  try
  {
    m_trained = m_classifier->train( trainData );
    if( m_trained )
    {
      m_featureDimension = features.cols;
    }
  }
  catch( const cv::Exception& e )
  {
    m_trained = false;
    return false;
  }

  return m_trained;
}

bool AdaBoostClassifier::trainFromProposals( const ObjectProposalVector& proposals,
                                              const std::vector< int >& labels )
{
  if( proposals.empty() || labels.empty() )
  {
    return false;
  }

  if( proposals.size() != labels.size() )
  {
    return false;
  }

  // Extract features from proposals
  int numSamples = static_cast< int >( proposals.size() );
  int featureDim = getProposalFeatureDimension();

  cv::Mat features( numSamples, featureDim, CV_32F );
  cv::Mat labelsMat( numSamples, 1, CV_32S );

  for( int i = 0; i < numSamples; i++ )
  {
    cv::Mat featureVec = extractFeatureVector( *proposals[i] );
    featureVec.copyTo( features.row( i ) );
    labelsMat.at< int >( i, 0 ) = labels[i];
  }

  return train( features, labelsMat );
}

int AdaBoostClassifier::classify( const cv::Mat& features ) const
{
  if( !m_trained || features.empty() )
  {
    return -1;
  }

  cv::Mat featuresFloat;
  if( features.type() != CV_32F )
  {
    features.convertTo( featuresFloat, CV_32F );
  }
  else
  {
    featuresFloat = features;
  }

  float response = m_classifier->predict( featuresFloat );
  return static_cast< int >( response );
}

int AdaBoostClassifier::classifyWithConfidence( const cv::Mat& features,
                                                 float& confidence ) const
{
  if( !m_trained || features.empty() )
  {
    confidence = 0.0f;
    return -1;
  }

  cv::Mat featuresFloat;
  if( features.type() != CV_32F )
  {
    features.convertTo( featuresFloat, CV_32F );
  }
  else
  {
    featuresFloat = features;
  }

  cv::Mat results;
  float response = m_classifier->predict( featuresFloat, results,
                                          cv::ml::StatModel::RAW_OUTPUT );

  confidence = results.at< float >( 0, 0 );
  return static_cast< int >( response );
}

void AdaBoostClassifier::classifyProposals( ObjectProposalVector& proposals,
                                             ObjectProposalVector& positives,
                                             float threshold ) const
{
  if( !m_trained )
  {
    return;
  }

  float useThreshold = ( threshold == -std::numeric_limits< float >::max() )
                       ? m_config.classificationThreshold
                       : threshold;

  for( auto& proposal : proposals )
  {
    if( !proposal->isActive )
    {
      continue;
    }

    cv::Mat features = extractFeatureVector( *proposal );
    float confidence;
    int prediction = classifyWithConfidence( features, confidence );

    if( confidence > useThreshold )
    {
      proposal->classification = static_cast< unsigned int >( prediction );
      proposal->classMagnitudes[0] = confidence;
      positives.push_back( proposal );
    }
  }
}

void AdaBoostClassifier::batchClassify( const cv::Mat& features,
                                         std::vector< int >& predictions,
                                         std::vector< float >& confidences ) const
{
  predictions.clear();
  confidences.clear();

  if( !m_trained || features.empty() )
  {
    return;
  }

  predictions.reserve( features.rows );
  confidences.reserve( features.rows );

  for( int i = 0; i < features.rows; i++ )
  {
    float confidence;
    int pred = classifyWithConfidence( features.row( i ), confidence );
    predictions.push_back( pred );
    confidences.push_back( confidence );
  }
}

bool AdaBoostClassifier::save( const std::string& filename ) const
{
  if( !m_trained || !m_classifier )
  {
    return false;
  }

  try
  {
    m_classifier->save( filename );

    // Save additional metadata
    std::string metaFile = filename + ".meta";
    cv::FileStorage fs( metaFile, cv::FileStorage::WRITE );
    fs << "featureDimension" << m_featureDimension;
    fs << "boostType" << static_cast< int >( m_config.boostType );
    fs << "classificationThreshold" << m_config.classificationThreshold;

    // Save class labels
    fs << "numClassLabels" << static_cast< int >( m_classLabels.size() );
    for( size_t i = 0; i < m_classLabels.size(); i++ )
    {
      std::string prefix = "label" + std::to_string( i );
      fs << ( prefix + "_id" ) << m_classLabels[i].id;
      fs << ( prefix + "_name" ) << m_classLabels[i].name;
      fs << ( prefix + "_desc" ) << m_classLabels[i].description;
    }

    fs.release();
  }
  catch( const cv::Exception& )
  {
    return false;
  }

  return true;
}

bool AdaBoostClassifier::load( const std::string& filename )
{
  try
  {
    m_classifier = cv::ml::Boost::load< cv::ml::Boost >( filename );

    // Load additional metadata
    std::string metaFile = filename + ".meta";
    cv::FileStorage fs( metaFile, cv::FileStorage::READ );
    if( fs.isOpened() )
    {
      fs["featureDimension"] >> m_featureDimension;

      int boostType;
      fs["boostType"] >> boostType;
      m_config.boostType = static_cast< AdaBoostType >( boostType );

      fs["classificationThreshold"] >> m_config.classificationThreshold;

      // Load class labels
      int numLabels;
      fs["numClassLabels"] >> numLabels;
      m_classLabels.clear();
      m_classLabels.reserve( numLabels );

      for( int i = 0; i < numLabels; i++ )
      {
        std::string prefix = "label" + std::to_string( i );
        ClassLabel label;
        fs[prefix + "_id"] >> label.id;
        fs[prefix + "_name"] >> label.name;
        fs[prefix + "_desc"] >> label.description;
        m_classLabels.push_back( label );
      }

      fs.release();
    }

    m_trained = m_classifier && !m_classifier->empty();
  }
  catch( const cv::Exception& )
  {
    m_trained = false;
    return false;
  }

  return m_trained;
}

bool AdaBoostClassifier::isTrained() const
{
  return m_trained;
}

int AdaBoostClassifier::getFeatureDimension() const
{
  return m_featureDimension;
}

void AdaBoostClassifier::setClassLabels( const std::vector< ClassLabel >& labels )
{
  m_classLabels = labels;
}

std::vector< ClassLabel > AdaBoostClassifier::getClassLabels() const
{
  return m_classLabels;
}

std::string AdaBoostClassifier::getClassName( int classId ) const
{
  for( const auto& label : m_classLabels )
  {
    if( label.id == classId )
    {
      return label.name;
    }
  }
  return "Unknown";
}

cv::Mat AdaBoostClassifier::extractFeatureVector( const ObjectProposal& proposal )
{
  int dim = getProposalFeatureDimension();
  cv::Mat features( 1, dim, CV_32F );

  int idx = 0;

  // Color features
  for( size_t i = 0; i < proposal.colorFeatures.size() && idx < dim; i++ )
  {
    features.at< float >( 0, idx++ ) = static_cast< float >( proposal.colorFeatures[i] );
  }

  // Gabor features
  for( size_t i = 0; i < proposal.gaborFeatures.size() && idx < dim; i++ )
  {
    features.at< float >( 0, idx++ ) = static_cast< float >( proposal.gaborFeatures[i] );
  }

  // Size features
  for( size_t i = 0; i < proposal.sizeFeatures.size() && idx < dim; i++ )
  {
    features.at< float >( 0, idx++ ) = static_cast< float >( proposal.sizeFeatures[i] );
  }

  // HoG features (flattened)
  for( size_t h = 0; h < proposal.hogResults.size(); h++ )
  {
    const cv::Mat& hog = proposal.hogResults[h];
    if( !hog.empty() )
    {
      for( int i = 0; i < hog.cols && idx < dim; i++ )
      {
        features.at< float >( 0, idx++ ) = hog.at< float >( 0, i );
      }
    }
  }

  return features;
}

int AdaBoostClassifier::getProposalFeatureDimension()
{
  // Color (122) + Gabor (36) + Size (9) + 2 * HoG (1764)
  return COLOR_FEATURES + GABOR_FEATURES + SIZE_FEATURES + NUM_HOG * HOG_FEATURES;
}

//------------------------------------------------------------------------------
// MultiClassAdaBoost Implementation
//------------------------------------------------------------------------------

MultiClassAdaBoost::MultiClassAdaBoost()
  : m_config()
  , m_numClasses( 0 )
  , m_trained( false )
{
}

MultiClassAdaBoost::MultiClassAdaBoost( const AdaBoostConfig& config )
  : m_config( config )
  , m_numClasses( 0 )
  , m_trained( false )
{
}

MultiClassAdaBoost::~MultiClassAdaBoost()
{
}

void MultiClassAdaBoost::setConfig( const AdaBoostConfig& config )
{
  m_config = config;
}

bool MultiClassAdaBoost::train( const cv::Mat& features,
                                 const cv::Mat& labels,
                                 int numClasses )
{
  if( features.empty() || labels.empty() || numClasses < 2 )
  {
    return false;
  }

  m_numClasses = numClasses;
  m_classifiers.clear();
  m_classifiers.resize( numClasses );

  // Convert features to float
  cv::Mat featuresFloat;
  if( features.type() != CV_32F )
  {
    features.convertTo( featuresFloat, CV_32F );
  }
  else
  {
    featuresFloat = features;
  }

  // Train one-vs-all classifiers
  for( int c = 0; c < numClasses; c++ )
  {
    // Create binary labels (1 for class c, 0 for others)
    cv::Mat binaryLabels( labels.rows, 1, CV_32S );
    for( int i = 0; i < labels.rows; i++ )
    {
      int label = labels.at< int >( i, 0 );
      binaryLabels.at< int >( i, 0 ) = ( label == c ) ? 1 : 0;
    }

    // Create training data
    cv::Ptr< cv::ml::TrainData > trainData = cv::ml::TrainData::create(
      featuresFloat, cv::ml::ROW_SAMPLE, binaryLabels );

    // Create and train classifier
    m_classifiers[c] = cv::ml::Boost::create();

    int boostType;
    switch( m_config.boostType )
    {
      case AdaBoostType::DISCRETE:
        boostType = cv::ml::Boost::DISCRETE;
        break;
      case AdaBoostType::REAL:
        boostType = cv::ml::Boost::REAL;
        break;
      case AdaBoostType::LOGIT:
        boostType = cv::ml::Boost::LOGIT;
        break;
      default:
        boostType = cv::ml::Boost::GENTLE;
    }

    m_classifiers[c]->setBoostType( boostType );
    m_classifiers[c]->setWeakCount( m_config.weakCount );
    m_classifiers[c]->setWeightTrimRate( m_config.weightTrimRate );
    m_classifiers[c]->setMaxDepth( m_config.maxDepth );

    try
    {
      if( !m_classifiers[c]->train( trainData ) )
      {
        m_trained = false;
        return false;
      }
    }
    catch( const cv::Exception& )
    {
      m_trained = false;
      return false;
    }
  }

  m_trained = true;
  return true;
}

int MultiClassAdaBoost::classify( const cv::Mat& features ) const
{
  std::vector< float > scores = getClassScores( features );

  if( scores.empty() )
  {
    return -1;
  }

  // Return class with highest score
  auto maxIt = std::max_element( scores.begin(), scores.end() );
  return static_cast< int >( std::distance( scores.begin(), maxIt ) );
}

std::vector< float > MultiClassAdaBoost::getClassScores( const cv::Mat& features ) const
{
  std::vector< float > scores;

  if( !m_trained || features.empty() )
  {
    return scores;
  }

  scores.resize( m_numClasses );

  cv::Mat featuresFloat;
  if( features.type() != CV_32F )
  {
    features.convertTo( featuresFloat, CV_32F );
  }
  else
  {
    featuresFloat = features;
  }

  for( int c = 0; c < m_numClasses; c++ )
  {
    cv::Mat results;
    m_classifiers[c]->predict( featuresFloat, results, cv::ml::StatModel::RAW_OUTPUT );
    scores[c] = results.at< float >( 0, 0 );
  }

  return scores;
}

bool MultiClassAdaBoost::save( const std::string& directory ) const
{
  if( !m_trained )
  {
    return false;
  }

  try
  {
    for( int c = 0; c < m_numClasses; c++ )
    {
      std::string filename = directory + "/classifier_" + std::to_string( c ) + ".xml";
      m_classifiers[c]->save( filename );
    }

    // Save metadata
    std::string metaFile = directory + "/meta.xml";
    cv::FileStorage fs( metaFile, cv::FileStorage::WRITE );
    fs << "numClasses" << m_numClasses;
    fs << "boostType" << static_cast< int >( m_config.boostType );
    fs << "weakCount" << m_config.weakCount;
    fs.release();
  }
  catch( const cv::Exception& )
  {
    return false;
  }

  return true;
}

bool MultiClassAdaBoost::load( const std::string& directory, int numClasses )
{
  try
  {
    // Load metadata first
    std::string metaFile = directory + "/meta.xml";
    cv::FileStorage fs( metaFile, cv::FileStorage::READ );
    if( fs.isOpened() )
    {
      fs["numClasses"] >> m_numClasses;

      int boostType;
      fs["boostType"] >> boostType;
      m_config.boostType = static_cast< AdaBoostType >( boostType );

      fs["weakCount"] >> m_config.weakCount;
      fs.release();
    }
    else
    {
      m_numClasses = numClasses;
    }

    m_classifiers.clear();
    m_classifiers.resize( m_numClasses );

    for( int c = 0; c < m_numClasses; c++ )
    {
      std::string filename = directory + "/classifier_" + std::to_string( c ) + ".xml";
      m_classifiers[c] = cv::ml::Boost::load< cv::ml::Boost >( filename );

      if( !m_classifiers[c] || m_classifiers[c]->empty() )
      {
        m_trained = false;
        return false;
      }
    }

    m_trained = true;
  }
  catch( const cv::Exception& )
  {
    m_trained = false;
    return false;
  }

  return m_trained;
}

bool MultiClassAdaBoost::isTrained() const
{
  return m_trained;
}

int MultiClassAdaBoost::getNumClasses() const
{
  return m_numClasses;
}

} // namespace opencv
} // namespace viame
