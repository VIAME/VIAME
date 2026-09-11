/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "iqr_session_adaboost.h"

#include <viame/algorithm_framework/logger/logger.h>

#include <cmath>

#ifdef VIAME_CORE_HAVE_PYTHON
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
#endif

namespace kv = kwiver::vital;

namespace viame
{

namespace iqr
{

namespace
{

constexpr const char* LOGGER_NAME = "viame.core.process_query_adaboost";

#ifdef VIAME_CORE_HAVE_PYTHON

// ----------------------------------------------------------------------------
/// Whether an interpreter exists to call into.
///
/// VIAME brings python up when the plugin loader loads `modules_python`, and
/// every pipeline that selects this process has it. A build or a tool that
/// does not is not an error here: the session reports no model and the base
/// class falls back to similarity, which is a working query.
bool
python_is_up()
{
  return Py_IsInitialized() != 0;
}

// ----------------------------------------------------------------------------
/// `viame.core.iqr_adaboost`, imported once.
///
/// Held rather than imported per call: importing scikit-learn is seconds,
/// and `predict_score` is called once per item in the working index, which
/// runs to tens of thousands.
py::module_&
backend()
{
  static py::module_ module = py::module_::import( "viame.core.iqr_adaboost" );
  return module;
}

#endif

} // anonymous namespace

// ============================================================================

class iqr_session_adaboost::priv
{
public:
  int m_weak_count = 100;
  int m_max_depth = 1;

  // Declared by the process, accepted, and no longer selecting anything.
  // Kept so a pipeline written for the OpenCV version still loads.
  std::string m_boost_type = "gentle";
  double m_weight_trim_rate = 0.95;

#ifdef VIAME_CORE_HAVE_PYTHON
  // The fitted estimator, or None
  py::object m_model;
#endif

  kv::logger_handle_t logger() const
  {
    return kv::get_logger( LOGGER_NAME );
  }

  /// One margin from the model, or zero when there is not one.
  double decision( const std::vector< double >& vec ) const
  {
#ifdef VIAME_CORE_HAVE_PYTHON
    if( !python_is_up() || !m_model || m_model.is_none() )
    {
      return 0.0;
    }

    try
    {
      py::gil_scoped_acquire gil;

      auto result = backend().attr( "decision" )(
        m_model, std::vector< std::vector< double > >{ vec } );

      auto values = result.cast< std::vector< double > >();

      return values.empty() ? 0.0 : values.front();
    }
    catch( const std::exception& e )
    {
      LOG_ERROR( logger(), "AdaBoost scoring failed: " << e.what() );
      return 0.0;
    }
#else
    ( void ) vec;
    return 0.0;
#endif
  }

  bool has_model() const
  {
#ifdef VIAME_CORE_HAVE_PYTHON
    return static_cast< bool >( m_model ) && !m_model.is_none();
#else
    return false;
#endif
  }
};

// ----------------------------------------------------------------------------
iqr_session_adaboost
::iqr_session_adaboost( unsigned pos_seed_neighbors )
  : iqr_session( pos_seed_neighbors )
  , d( new priv() )
{}

// ----------------------------------------------------------------------------
iqr_session_adaboost
::~iqr_session_adaboost()
{
#ifdef VIAME_CORE_HAVE_PYTHON
  // The held `py::object` needs the GIL to drop its reference, and the
  // destructor can run on any thread
  if( python_is_up() && d && d->m_model )
  {
    py::gil_scoped_acquire gil;
    d->m_model = py::object();
  }
#endif
}

// ----------------------------------------------------------------------------
void
iqr_session_adaboost
::set_boost_type( const std::string& type )
{
  if( type != d->m_boost_type )
  {
    LOG_INFO( d->logger(), "boost_type is '" << type << "'; scikit-learn "
      "implements SAMME only, so the setting is accepted and ignored. It "
      "chose between OpenCV's DISCRETE, REAL, LOGIT and GENTLE." );
  }

  d->m_boost_type = type;
}

// ----------------------------------------------------------------------------
void
iqr_session_adaboost
::set_weak_count( int count )
{
  d->m_weak_count = count;
}

// ----------------------------------------------------------------------------
void
iqr_session_adaboost
::set_max_depth( int depth )
{
  d->m_max_depth = depth;
}

// ----------------------------------------------------------------------------
void
iqr_session_adaboost
::set_weight_trim_rate( double rate )
{
  if( rate != d->m_weight_trim_rate )
  {
    LOG_INFO( d->logger(), "weight_trim_rate is " << rate << "; scikit-learn "
      "does not trim sample weights, so the setting is accepted and "
      "ignored." );
  }

  d->m_weight_trim_rate = rate;
}

// ----------------------------------------------------------------------------
bool
iqr_session_adaboost
::is_model_valid() const
{
  return d->has_model();
}

// ----------------------------------------------------------------------------
void
iqr_session_adaboost
::free_model()
{
#ifdef VIAME_CORE_HAVE_PYTHON
  if( python_is_up() && d->m_model )
  {
    py::gil_scoped_acquire gil;
    d->m_model = py::object();
  }
#endif
}

// ----------------------------------------------------------------------------
double
iqr_session_adaboost
::predict_score( const std::vector< double >& vec ) const
{
  if( !is_model_valid() )
  {
    return compute_positive_similarity( vec );
  }

  // The margin through a logistic, the way the OpenCV path did it, so the
  // score stays in [0, 1] and the process's thresholds still mean what they
  // meant. What is different is that the margin is now a real number rather
  // than a class label, so this is graded rather than two-valued.
  return 1.0 / ( 1.0 + std::exp( -d->decision( vec ) ) );
}

// ----------------------------------------------------------------------------
double
iqr_session_adaboost
::predict_distance( const std::vector< double >& vec ) const
{
  if( !is_model_valid() )
  {
    return 0.0;
  }

  return d->decision( vec );
}

// ----------------------------------------------------------------------------
std::vector< unsigned char >
iqr_session_adaboost
::get_model_bytes() const
{
#ifdef VIAME_CORE_HAVE_PYTHON
  if( !is_model_valid() || !python_is_up() )
  {
    return {};
  }

  try
  {
    py::gil_scoped_acquire gil;

    auto result_to_string = []( py::object value )
    {
      return value.cast< std::string >();
    };

    // `py::bytes` casts to `std::string`, not to a vector of bytes: the
    // stl caster sees a sequence of length-one bytes objects otherwise and
    // refuses the lot
    const auto blob = result_to_string(
      backend().attr( "dumps" )( d->m_model ) );

    return std::vector< unsigned char >( blob.begin(), blob.end() );
  }
  catch( const std::exception& e )
  {
    LOG_ERROR( d->logger(), "Saving the AdaBoost model failed: " << e.what() );
    return {};
  }
#else
  return {};
#endif
}

// ----------------------------------------------------------------------------
bool
iqr_session_adaboost
::load_model_from_bytes( const std::vector< unsigned char >& bytes )
{
  free_model();

  if( bytes.empty() )
  {
    return false;
  }

#ifdef VIAME_CORE_HAVE_PYTHON
  if( !python_is_up() )
  {
    LOG_ERROR( d->logger(), "Cannot load an AdaBoost model without python." );
    return false;
  }

  try
  {
    py::gil_scoped_acquire gil;

    d->m_model = backend().attr( "loads" )( py::bytes(
      reinterpret_cast< const char* >( bytes.data() ), bytes.size() ) );
  }
  catch( const std::exception& e )
  {
    LOG_ERROR( d->logger(), "Loading the AdaBoost model failed: "
      << e.what() );
    return false;
  }

  return is_model_valid();
#else
  LOG_ERROR( d->logger(), "Cannot load an AdaBoost model: this build has no "
    "python." );
  return false;
#endif
}

// ----------------------------------------------------------------------------
std::string
iqr_session_adaboost
::logger_name() const
{
  return LOGGER_NAME;
}

// ----------------------------------------------------------------------------
bool
iqr_session_adaboost
::train_model( const std::vector< descriptor_element >& auto_negatives )
{
  const size_t positives = m_positive_descriptors.size();
  const size_t negatives = m_negative_descriptors.size() + auto_negatives.size();

  LOG_INFO( d->logger(), "AdaBoost training: " << positives << " positives, "
    << negatives << " negatives (" << auto_negatives.size()
    << " auto-negatives)" );

  if( positives + negatives < 2 || positives == 0 || negatives == 0 )
  {
    return false;
  }

#ifdef VIAME_CORE_HAVE_PYTHON
  if( !python_is_up() )
  {
    LOG_ERROR( d->logger(), "Cannot train an AdaBoost model without python; "
      "the query falls back to similarity against its positives." );
    return false;
  }

  // Positives first and then negatives, which is the order the OpenCV
  // version built its matrix in. It does not change what is learned, but it
  // keeps the two readable against each other.
  std::vector< std::vector< double > > features;
  std::vector< int > labels;

  features.reserve( positives + negatives );
  labels.reserve( positives + negatives );

  for( const auto& entry : m_positive_descriptors )
  {
    features.push_back( entry.second.vector );
    labels.push_back( 1 );
  }

  for( const auto& entry : m_negative_descriptors )
  {
    features.push_back( entry.second.vector );
    labels.push_back( 0 );
  }

  for( const auto& entry : auto_negatives )
  {
    features.push_back( entry.vector );
    labels.push_back( 0 );
  }

  // A descriptor of a different length than the rest would reach numpy as a
  // ragged array, which it refuses in a way that says nothing useful
  const size_t dimension = features.front().size();

  for( const auto& vector : features )
  {
    if( vector.size() != dimension )
    {
      LOG_ERROR( d->logger(), "Descriptors are not all the same length ("
        << dimension << " and " << vector.size() << "); cannot train." );
      return false;
    }
  }

  try
  {
    py::gil_scoped_acquire gil;

    d->m_model = backend().attr( "train" )(
      features, labels, d->m_weak_count, d->m_max_depth );
  }
  catch( const std::exception& e )
  {
    LOG_ERROR( d->logger(), "AdaBoost training failed: " << e.what() );
    free_model();
    return false;
  }

  if( !is_model_valid() )
  {
    LOG_WARN( d->logger(), "AdaBoost training produced no model." );
    return false;
  }

  return true;
#else
  ( void ) auto_negatives;
  LOG_ERROR( d->logger(), "Cannot train an AdaBoost model: this build has no "
    "python." );
  return false;
#endif
}

} // end namespace iqr
} // end namespace viame
