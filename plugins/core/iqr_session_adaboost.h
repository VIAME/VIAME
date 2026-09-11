/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief AdaBoost IQR session, backed by scikit-learn.
 */

#ifndef VIAME_CORE_IQR_SESSION_ADABOOST_H
#define VIAME_CORE_IQR_SESSION_ADABOOST_H

#include "iqr_session.h"

#include "viame_core_export.h"

#include <memory>
#include <string>
#include <vector>

namespace viame
{

namespace iqr
{

// ----------------------------------------------------------------------------
/// The IQR session `process_query_adaboost` trains and ranks with.
///
/// `cv::ml::Boost` until P7-T09, `sklearn.ensemble.AdaBoostClassifier` since.
/// The model lives in python -- `viame.core.iqr_adaboost` -- and this is the
/// C++ side of the four calls that were the model: fit, score, save, load.
/// Everything else about an IQR session, which is most of it, is the base
/// class and did not move.
///
/// Two of the four configuration keys no longer select anything, and are
/// kept rather than dropped so that a pipeline written for the OpenCV
/// version still loads and the registry baseline still holds. `boost_type`
/// chose between DISCRETE, REAL, LOGIT and GENTLE, and scikit-learn has only
/// SAMME; `weight_trim_rate` was OpenCV's sample-weight trimming, which it
/// has no equivalent of. Both are logged once when set to something other
/// than their default, so a user who was relying on them is told.
///
/// Without python -- a build with `VIAME_ENABLE_PYTHON` off, or an install
/// without scikit-learn -- `train_model` fails and the session falls back to
/// similarity scoring against the positive exemplars, which is what the base
/// class does whenever there is no model. The query still returns results.
class VIAME_CORE_EXPORT iqr_session_adaboost : public iqr_session
{
public:
  explicit iqr_session_adaboost( unsigned pos_seed_neighbors );

  ~iqr_session_adaboost() override;

  iqr_session_adaboost( const iqr_session_adaboost& ) = delete;
  iqr_session_adaboost& operator=( const iqr_session_adaboost& ) = delete;

  // -- AdaBoost-specific config setters --

  /// Accepted and ignored: scikit-learn implements SAMME only.
  void set_boost_type( const std::string& type );

  /// `n_estimators`.
  void set_weak_count( int count );

  /// The depth of each weak learner; one is a decision stump.
  void set_max_depth( int depth );

  /// Accepted and ignored: scikit-learn does not trim sample weights.
  void set_weight_trim_rate( double rate );

  // -- Virtual interface implementation --

  bool is_model_valid() const override;
  void free_model() override;
  double predict_score( const std::vector< double >& vec ) const override;
  double predict_distance( const std::vector< double >& vec ) const override;
  std::vector< unsigned char > get_model_bytes() const override;
  bool load_model_from_bytes(
    const std::vector< unsigned char >& bytes ) override;

protected:
  std::string logger_name() const override;

  bool train_model(
    const std::vector< descriptor_element >& auto_negatives ) override;

private:
  class priv;
  std::unique_ptr< priv > d;
};

} // end namespace iqr
} // end namespace viame

#endif // VIAME_CORE_IQR_SESSION_ADABOOST_H
