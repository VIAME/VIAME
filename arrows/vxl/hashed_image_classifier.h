// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_ARROWS_VXL_HASHED_IMAGE_CLASSIFIER_
#define KWIVER_ARROWS_VXL_HASHED_IMAGE_CLASSIFIER_

#include <vital/logger/logger.h>

#include <vgl/vgl_box_2d.h>
#include <vil/vil_image_view.h>
#include <vil/vil_rgb.h>

#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

namespace kwiver {

namespace arrows {

namespace vxl {

template < typename FeatureType, typename OutputType >
class hashed_image_classifier;

/// Internal data required for the hashed_image_classifier class.
///
/// The data members of this class are exposed just in case any external
/// functions or processes want to train their own models, usually in an
/// online fashion.
template < typename FloatType = double >
class hashed_image_classifier_model
{
public:
  using self_t = hashed_image_classifier_model< FloatType >;
  using weight_t = FloatType;

  hashed_image_classifier_model() = default;
  hashed_image_classifier_model( self_t const& other );

  virtual ~hashed_image_classifier_model() = default;

  // Number of features
  unsigned int num_features = 0;

  // Maximum value for each feature
  std::vector< unsigned > max_feature_value;

  // Root pointer to start of weight array for all features
  std::vector< weight_t > weights;

  // Pointers to the start of each individual feature weight vector
  std::vector< weight_t* > feature_weights;

  // Is the current classifier configured correctly?
  bool is_valid() const;

  // Copy operator
  self_t& operator=( self_t const& other );

  // Seed the model with empty values of the given size
  void reset( unsigned feature_count, unsigned entries_per_feature );

  // Normalize internal histogram weights to a given absolute sum
  void normalize( weight_t total_weight = 1.0 );

private:
  vital::logger_handle_t logger =
    kwiver::vital::get_logger( "arrows.vxl.hashed_image_classifier" );
};

/// Stream operator declaration for the hashed_image_classifier model class.
template < typename FloatType > std::ostream&
operator<<( std::ostream& os,
            hashed_image_classifier_model< FloatType > const& obj );

/// Stream operator declaration for the hashed_image_classifier class.
template < typename FeatureType, typename OutputType >
std::ostream&
operator<<( std::ostream& os,
            hashed_image_classifier< FeatureType, OutputType > const& obj );

/// A classifier designed to efficiently classify every pixel in an image.
///
/// Every feature (for every pixel) is required to be mapped onto unsigned
/// integral type HashType before input. For every possible value of this
/// hashed mapping, and for every feature, a single weight must be given (in a
/// model file) which contributes to some binary decision about said pixel.
///
/// More formally, for every feature at location i,j, i.e. f(i,j,0), f(i,j,1)
/// f(i,j,...) the output classification is given as:
///
///                      c(i,j) = sum( w_{k}[ f(i,j,k) ] )
///
///       for every feature k, and some corresponding weight vector w_{k}
///
/// Typically these models may be specified manually, created with the help of
/// constructs such as parzen window estimates, or learned via supervised
/// learning techniques such as using boosted linear discriminants. Compared
/// against other modern classifiers, it places significant constraints on how
/// features can be used together in favor of efficiency.

template < typename FeatureType, typename OutputType = double >
class hashed_image_classifier
{
public:
  using weight_t = OutputType;
  using weight_image_t = vil_image_view< weight_t >;
  using mask_image_t = vil_image_view< bool >;
  using input_t = FeatureType;
  using input_image_t = vil_image_view< input_t >;
  using feature_vector_t = std::vector< input_image_t >;
  using model_t = hashed_image_classifier_model< OutputType >;
  using model_sptr_t = std::shared_ptr< model_t >;
  using self_t = hashed_image_classifier< FeatureType, OutputType >;

  /// Default constructor; a model must be loaded via load_from_file before
  /// use.
  hashed_image_classifier() : model_( new model_t() ) {}

  virtual ~hashed_image_classifier() = default;

  /// Load a model from a file.
  ///
  /// The model file should be specified in the following format, where
  /// bracketed items should be replaced by values:
  ///
  /// [Number of features]
  /// [Number of values feature 0 can take on] [weights for value 0, 1, etc. ]
  /// [Number of values feature 1 can take on] [weights for value 0, 1, etc. ]
  /// etc...
  virtual bool load_from_file( std::string const& file );

  /// Classify a feature array, in addition to adding offset to each pixel.
  virtual void classify_images( input_image_t const& input_features,
                                weight_image_t& output_image,
                                weight_t const offset = 0.0 ) const;

  /// Classify a feature array, in addition to adding offset to each pixel.
  virtual void classify_images( feature_vector_t const& input_features,
                                weight_image_t& output_image,
                                weight_t const offset = 0.0 ) const;

  /// Only classify certain pixels as given by a mask.
  virtual void classify_images( feature_vector_t const& input_features,
                                mask_image_t const& mask,
                                weight_image_t& output_image,
                                weight_t const offset = 0.0 ) const;

  /// Generate a weight image for some feature.
  virtual void generate_weight_image( input_image_t const& src,
                                      weight_image_t& dst,
                                      unsigned const& feature_id ) const;

  /// Returns the number of features the loaded model contains info for.
  virtual unsigned
  feature_count() const { return model_->num_features; }

  /// Was a valid model loaded by this classifier.
  virtual bool
  is_valid() const { return model_ && model_->is_valid(); }

  /// Set the internal model from some external source.
  virtual void set_model( model_sptr_t external_model );

  /// The stream operator function for writing out models.
  friend std::ostream& operator<<<>( std::ostream& os, self_t const& obj );

protected:
  // A pointer to our internal data
  model_sptr_t model_;
  vital::logger_handle_t logger =
    kwiver::vital::get_logger( "arrows.vxl.hashed_image_classifier" );
};

} // namespace vxl

} // namespace arrows

} // namespace kwiver

#endif
