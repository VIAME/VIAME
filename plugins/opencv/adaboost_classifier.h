/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_OPENCV_ADABOOST_CLASSIFIER_H
#define VIAME_OPENCV_ADABOOST_CLASSIFIER_H

#include "viame_opencv_export.h"
#include "ellipse_proposal.h"

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>

#include <memory>
#include <string>
#include <vector>

namespace viame {
namespace opencv {

// AdaBoost algorithm type
enum class AdaBoostType
{
  DISCRETE = 0,  // Discrete AdaBoost (standard)
  REAL     = 1,  // Real AdaBoost
  LOGIT    = 2,  // LogitBoost
  GENTLE   = 3   // Gentle AdaBoost
};

// AdaBoost classifier configuration
struct VIAME_OPENCV_EXPORT AdaBoostConfig
{
  // Boosting algorithm type
  AdaBoostType boostType = AdaBoostType::GENTLE;

  // Number of weak classifiers (boosting iterations)
  int weakCount = 100;

  // Weight trim rate (0-1, higher = faster but less accurate)
  double weightTrimRate = 0.95;

  // Maximum depth of weak classifier decision trees
  int maxDepth = 1;

  // Use surrogates for missing data handling
  bool useSurrogates = false;

  // Minimum samples per node
  int minSampleCount = 10;

  // Classification threshold (predictions above this are positive)
  float classificationThreshold = 0.0f;
};

// Class label information
struct ClassLabel
{
  int id;
  std::string name;
  std::string description;
};

// AdaBoost Classifier for Object Proposals
//
// This class wraps OpenCV's cv::ml::Boost to provide AdaBoost classification
// for object proposals based on extracted features (HoG, Gabor, color, etc.).
//
// It supports:
// - Training from feature vectors
// - Multi-class classification via one-vs-all
// - Model persistence (save/load)
// - Probability estimation
class VIAME_OPENCV_EXPORT AdaBoostClassifier
{
public:
  AdaBoostClassifier();
  explicit AdaBoostClassifier( const AdaBoostConfig& config );
  ~AdaBoostClassifier();

  // Configuration
  void setConfig( const AdaBoostConfig& config );
  AdaBoostConfig getConfig() const;

  // Training
  //
  // @param features Matrix of feature vectors (rows = samples, cols = features)
  // @param labels Vector of class labels (0 = negative, 1+ = positive class)
  // @return true on success
  bool train( const cv::Mat& features, const cv::Mat& labels );

  // Train from ObjectProposals (extracts features internally)
  //
  // @param proposals Training proposals with features already extracted
  // @param labels Class labels for each proposal
  // @return true on success
  bool trainFromProposals( const ObjectProposalVector& proposals,
                           const std::vector< int >& labels );

  // Classify a single feature vector
  //
  // @param features Feature vector (1 x numFeatures)
  // @return Predicted class label
  int classify( const cv::Mat& features ) const;

  // Classify with confidence score
  //
  // @param features Feature vector
  // @param confidence Output confidence (sum of weak classifier votes)
  // @return Predicted class label
  int classifyWithConfidence( const cv::Mat& features, float& confidence ) const;

  // Classify all proposals
  //
  // @param proposals Input proposals with features extracted
  // @param positives Output proposals classified as positive
  // @param threshold Classification threshold (default uses config)
  void classifyProposals( ObjectProposalVector& proposals,
                          ObjectProposalVector& positives,
                          float threshold = -std::numeric_limits< float >::max() ) const;

  // Batch classify
  //
  // @param features Matrix of feature vectors
  // @param predictions Output predictions
  // @param confidences Output confidence scores
  void batchClassify( const cv::Mat& features,
                      std::vector< int >& predictions,
                      std::vector< float >& confidences ) const;

  // Model persistence
  bool save( const std::string& filename ) const;
  bool load( const std::string& filename );

  // Query
  bool isTrained() const;
  int getFeatureDimension() const;

  // Class label management
  void setClassLabels( const std::vector< ClassLabel >& labels );
  std::vector< ClassLabel > getClassLabels() const;
  std::string getClassName( int classId ) const;

  // Feature extraction helper
  //
  // Extract a feature vector from a proposal's stored features
  // Order: colorFeatures, gaborFeatures, sizeFeatures, hogResults (flattened)
  static cv::Mat extractFeatureVector( const ObjectProposal& proposal );

  // Get expected feature dimension for a proposal
  static int getProposalFeatureDimension();

private:
  // Convert OpenCV boost type
  int getOpenCVBoostType() const;

  // Create and configure the boost classifier
  cv::Ptr< cv::ml::Boost > createBoostClassifier() const;

  AdaBoostConfig m_config;
  cv::Ptr< cv::ml::Boost > m_classifier;
  std::vector< ClassLabel > m_classLabels;
  int m_featureDimension;
  bool m_trained;
};

// Multi-class AdaBoost using one-vs-all strategy
//
// Trains separate binary classifiers for each class
class VIAME_OPENCV_EXPORT MultiClassAdaBoost
{
public:
  MultiClassAdaBoost();
  explicit MultiClassAdaBoost( const AdaBoostConfig& config );
  ~MultiClassAdaBoost();

  // Set configuration for all classifiers
  void setConfig( const AdaBoostConfig& config );

  // Train multi-class classifier
  //
  // @param features Feature matrix
  // @param labels Class labels (0 to numClasses-1)
  // @param numClasses Number of classes
  // @return true on success
  bool train( const cv::Mat& features, const cv::Mat& labels, int numClasses );

  // Classify with winner-takes-all
  int classify( const cv::Mat& features ) const;

  // Get confidence scores for all classes
  std::vector< float > getClassScores( const cv::Mat& features ) const;

  // Model persistence
  bool save( const std::string& directory ) const;
  bool load( const std::string& directory, int numClasses );

  bool isTrained() const;
  int getNumClasses() const;

private:
  AdaBoostConfig m_config;
  std::vector< cv::Ptr< cv::ml::Boost > > m_classifiers;
  int m_numClasses;
  bool m_trained;
};

} // namespace opencv
} // namespace viame

#endif /* VIAME_OPENCV_ADABOOST_CLASSIFIER_H */
