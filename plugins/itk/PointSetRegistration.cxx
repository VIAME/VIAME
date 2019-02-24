#include "itkMeshFileReader.h"
#include "itkMesh.h"
#include "itkEuclideanDistancePointSetToPointSetMetricv4.h"
#include "itkExpectationBasedPointSetToPointSetMetricv4.h"
#include "itkJensenHavrdaCharvatTsallisPointSetToPointSetMetricv4.h"
#include "itkGradientDescentOptimizerv4.h"
#include "itkRegistrationParameterScalesFromPhysicalShift.h"
#include "itkAffineTransform.h"
#include "itkCommand.h"
#include "itkTransformFileWriter.h"

template< typename TFilter >
class RegistrationIterationUpdateCommand: public itk::Command
{
public:
  using Self = RegistrationIterationUpdateCommand;;

  using Superclass = itk::Command;
  using Pointer = itk::SmartPointer<Self>;
  itkNewMacro( Self );

protected:
  RegistrationIterationUpdateCommand() = default;

public:

  void Execute(itk::Object *caller, const itk::EventObject & event) override
    {
    Execute( (const itk::Object *) caller, event);
    }

  void Execute(const itk::Object * object, const itk::EventObject & event) override
    {
    if( typeid( event ) != typeid( itk::IterationEvent ) )
      {
      return;
      }
    const auto * optimizer = dynamic_cast< const TFilter * >( object );

    if( !optimizer )
      {
      itkGenericExceptionMacro( "Error dynamic_cast failed" );
      }
    std::cout << "It: " << optimizer->GetCurrentIteration()
              << " metric value: " << optimizer->GetCurrentMetricValue()
              << " position: " << optimizer->GetCurrentPosition();

    std::cout << std::endl;
    }
};

template< typename TTransform, typename TMetric, typename TPointSet >
int EuclideanDistancePointSetMetricRegistration(
  unsigned int numberOfIterations, double maximumPhysicalStepSize,
  typename TTransform::Pointer & transform, typename TMetric::Pointer & metric,
  typename TPointSet::Pointer & fixedPoints, typename TPointSet::Pointer & movingPoints )
{
  using PointSetType = TPointSet;
  using PointType = typename PointSetType::PointType;
  using CoordRepType = typename PointType::CoordRepType;

  // Finish setting up the metric
  metric->SetFixedPointSet( fixedPoints );
  metric->SetMovingPointSet( movingPoints );
  metric->SetMovingTransform( transform );
  metric->Initialize();

  // scales estimator
  using RegistrationParameterScalesFromShiftType =
    itk::RegistrationParameterScalesFromPhysicalShift< TMetric >;
  typename RegistrationParameterScalesFromShiftType::Pointer shiftScaleEstimator =
    RegistrationParameterScalesFromShiftType::New();
  shiftScaleEstimator->SetMetric( metric );
  // needed with pointset metrics
  shiftScaleEstimator->SetVirtualDomainPointSet( metric->GetVirtualTransformedPointSet() );

  // optimizer
  using OptimizerType = itk::GradientDescentOptimizerv4;
  typename OptimizerType::Pointer optimizer = OptimizerType::New();
  optimizer->SetMetric( metric );
  optimizer->SetNumberOfIterations( numberOfIterations );
  optimizer->SetScalesEstimator( shiftScaleEstimator );
  optimizer->SetMaximumStepSizeInPhysicalUnits( maximumPhysicalStepSize );

  using CommandType = RegistrationIterationUpdateCommand<OptimizerType>;
  typename CommandType::Pointer observer = CommandType::New();
  optimizer->AddObserver( itk::IterationEvent(), observer );

  std::cout << "Transform" << *transform << std::endl;
  // start
  optimizer->StartOptimization();

  std::cout << "numberOfIterations: " << numberOfIterations << std::endl;
  std::cout << "maximumPhysicalStepSize: " << maximumPhysicalStepSize << std::endl;
  std::cout << "Optimizer scales: " << optimizer->GetScales() << std::endl;
  std::cout << "Optimizer learning rate: " << optimizer->GetLearningRate() << std::endl;
  std::cout << "Moving-source final value: " << optimizer->GetCurrentMetricValue() << std::endl;
  if( transform->GetTransformCategory() == TTransform::DisplacementField )
    {
    std::cout << "local-support transform non-zero parameters: " << std::endl;
    typename TTransform::ParametersType params = transform->GetParameters();
    for( itk::SizeValueType n = 0; n < transform->GetNumberOfParameters();
         n += transform->GetNumberOfLocalParameters() )
      {
      typename TTransform::ParametersValueType zero =
        itk::NumericTraits<typename TTransform::ParametersValueType>::ZeroValue();
      if( itk::Math::NotExactlyEquals(params[n], zero) && itk::Math::NotExactlyEquals(params[n+1], zero) )
        {
        std::cout << n << ", " << n+1 << " : " << params[n] << ", " << params[n+1] << std::endl;
        }
      }
    }
  else
    {
    std::cout << "Moving-source final position: " << optimizer->GetCurrentPosition() << std::endl;
    }
  std::cout << "Transform" << *transform << std::endl;

  return EXIT_SUCCESS;
}

template< typename TTransform, typename TMetric, typename TPointSet >
int ExpectationBasedPointSetMetricRegistration(
  unsigned int numberOfIterations, double maximumPhysicalStepSize,
  typename TTransform::Pointer & transform, typename TMetric::Pointer & metric,
  typename TPointSet::Pointer & fixedPoints, typename TPointSet::Pointer & movingPoints,
  double pointSetSigma )
{
  using PointSetType = TPointSet;
  using PointType = typename PointSetType::PointType;
  using CoordRepType = typename PointType::CoordRepType;

  // Finish setting up the metric
  metric->SetFixedPointSet( fixedPoints );
  metric->SetMovingPointSet( movingPoints );
  metric->SetMovingTransform( transform );
  metric->SetPointSetSigma( pointSetSigma );
  metric->SetEvaluationKNeighborhood( 50 );
  metric->Initialize();

  // scales estimator
  using RegistrationParameterScalesFromShiftType =
    itk::RegistrationParameterScalesFromPhysicalShift< TMetric >;
  typename RegistrationParameterScalesFromShiftType::Pointer shiftScaleEstimator =
    RegistrationParameterScalesFromShiftType::New();
  shiftScaleEstimator->SetMetric( metric );
  // needed with pointset metrics
  shiftScaleEstimator->SetVirtualDomainPointSet( metric->GetVirtualTransformedPointSet() );

  // optimizer
  using OptimizerType = itk::GradientDescentOptimizerv4;
  typename OptimizerType::Pointer optimizer = OptimizerType::New();
  optimizer->SetMetric( metric );
  optimizer->SetNumberOfIterations( numberOfIterations );
  optimizer->SetScalesEstimator( shiftScaleEstimator );
  optimizer->SetMaximumStepSizeInPhysicalUnits( maximumPhysicalStepSize );

  using CommandType = RegistrationIterationUpdateCommand<OptimizerType>;
  typename CommandType::Pointer observer = CommandType::New();
  optimizer->AddObserver( itk::IterationEvent(), observer );

  std::cout << "Transform" << *transform << std::endl;
  // start
  optimizer->StartOptimization();

  std::cout << "numberOfIterations: " << numberOfIterations << std::endl;
  std::cout << "maximumPhysicalStepSize: " << maximumPhysicalStepSize << std::endl;
  std::cout << "Optimizer scales: " << optimizer->GetScales() << std::endl;
  std::cout << "Optimizer learning rate: " << optimizer->GetLearningRate() << std::endl;
  std::cout << "Moving-source final value: " << optimizer->GetCurrentMetricValue() << std::endl;
  if( transform->GetTransformCategory() == TTransform::DisplacementField )
    {
    std::cout << "local-support transform non-zero parameters: " << std::endl;
    typename TTransform::ParametersType params = transform->GetParameters();
    for( itk::SizeValueType n = 0; n < transform->GetNumberOfParameters();
         n += transform->GetNumberOfLocalParameters() )
      {
      typename TTransform::ParametersValueType zero =
        itk::NumericTraits<typename TTransform::ParametersValueType>::ZeroValue();
      if( itk::Math::NotExactlyEquals(params[n], zero) && itk::Math::NotExactlyEquals(params[n+1], zero) )
        {
        std::cout << n << ", " << n+1 << " : " << params[n] << ", " << params[n+1] << std::endl;
        }
      }
    }
  else
    {
    std::cout << "Moving-source final position: " << optimizer->GetCurrentPosition() << std::endl;
    }
  std::cout << "Transform" << *transform << std::endl;

  return EXIT_SUCCESS;
}

template< typename TTransform, typename TMetric, typename TPointSet >
int JHCTPointSetMetricRegistration(
  unsigned int numberOfIterations, double maximumPhysicalStepSize,
  typename TTransform::Pointer & transform, typename TMetric::Pointer & metric,
  typename TPointSet::Pointer & fixedPoints, typename TPointSet::Pointer & movingPoints,
  double pointSetSigma )
{
  using PointSetType = TPointSet;
  using PointType = typename PointSetType::PointType;
  using CoordRepType = typename PointType::CoordRepType;

  // Finish setting up the metric
  metric->SetFixedPointSet( fixedPoints );
  metric->SetMovingPointSet( movingPoints );
  metric->SetMovingTransform( transform );
  metric->SetPointSetSigma( pointSetSigma );
  metric->SetEvaluationKNeighborhood( 50 );
  metric->SetUseAnisotropicCovariances( true );
  metric->SetAlpha( 1.1 );
  metric->Initialize();

  // scales estimator
  using RegistrationParameterScalesFromShiftType =
    itk::RegistrationParameterScalesFromPhysicalShift< TMetric >;
  typename RegistrationParameterScalesFromShiftType::Pointer shiftScaleEstimator =
    RegistrationParameterScalesFromShiftType::New();
  shiftScaleEstimator->SetMetric( metric );
  // needed with pointset metrics
  shiftScaleEstimator->SetVirtualDomainPointSet( metric->GetVirtualTransformedPointSet() );

  // optimizer
  using OptimizerType = itk::GradientDescentOptimizerv4;
  typename OptimizerType::Pointer optimizer = OptimizerType::New();
  optimizer->SetMetric( metric );
  optimizer->SetNumberOfIterations( numberOfIterations );
  optimizer->SetScalesEstimator( shiftScaleEstimator );
  optimizer->SetMaximumStepSizeInPhysicalUnits( maximumPhysicalStepSize );

  using CommandType = RegistrationIterationUpdateCommand<OptimizerType>;
  typename CommandType::Pointer observer = CommandType::New();
  optimizer->AddObserver( itk::IterationEvent(), observer );

  std::cout << "Transform" << *transform << std::endl;
  // start
  optimizer->StartOptimization();

  std::cout << "numberOfIterations: " << numberOfIterations << std::endl;
  std::cout << "maximumPhysicalStepSize: " << maximumPhysicalStepSize << std::endl;
  std::cout << "Optimizer scales: " << optimizer->GetScales() << std::endl;
  std::cout << "Optimizer learning rate: " << optimizer->GetLearningRate() << std::endl;
  std::cout << "Moving-source final value: " << optimizer->GetCurrentMetricValue() << std::endl;
  if( transform->GetTransformCategory() == TTransform::DisplacementField )
    {
    std::cout << "local-support transform non-zero parameters: " << std::endl;
    typename TTransform::ParametersType params = transform->GetParameters();
    for( itk::SizeValueType n = 0; n < transform->GetNumberOfParameters(); n += transform->GetNumberOfLocalParameters() )
      {
      typename TTransform::ParametersValueType zero = itk::NumericTraits<typename TTransform::ParametersValueType>::ZeroValue();
      if( itk::Math::NotExactlyEquals(params[n], zero) && itk::Math::NotExactlyEquals(params[n+1], zero) )
        {
        std::cout << n << ", " << n+1 << " : " << params[n] << ", " << params[n+1] << std::endl;
        }
      }
    }
  else
    {
    std::cout << "Moving-source final position: " << optimizer->GetCurrentPosition() << std::endl;
    }
  std::cout << "Transform" << *transform << std::endl;

  return EXIT_SUCCESS;
}
int main(int argc, char * argv[])
{
  if( argc < 4 )
    {
    std::cerr << "Usage: " << argv[0] << " <InputFixedMesh> <InputMovingMesh> <OutputTransform> "
              << "<OutputTransformedFixedMesh> [MetricId] [NumberOfIterations] "
              << "[MaximumPhysicalStepSize] [PointSetSigma]" << std::endl;

    return EXIT_FAILURE;
    }

  const char * inputFixedMeshFile = argv[1];
  const char * inputMovingMeshFile = argv[2];
  const char * outputTransformFile = argv[3];

  unsigned int metricId = 2;
  unsigned int numberOfIterations = 100;
  auto maximumPhysicalStepSize = static_cast<double>( 2.0 );
  double pointSetSigma = 3.0;

  if( argc > 4 )
    {
    metricId = std::stoi( argv[4] );
    }
  if( argc > 5 )
    {
    numberOfIterations = std::stoi( argv[5] );
    }
  if( argc > 6 )
    {
    maximumPhysicalStepSize = std::stod( argv[6] );
    }
  if( argc > 7 )
    {
    pointSetSigma = std::stod( argv[7] );
    }

  constexpr unsigned int Dimension = 2;
  using PointSetType = itk::PointSet<unsigned char, Dimension>;
  using MeshType = itk::Mesh<unsigned char, Dimension>;

  using MeshReaderType = itk::MeshFileReader< MeshType >;
  MeshReaderType::Pointer fixedReader = MeshReaderType::New();
  fixedReader->SetFileName( inputFixedMeshFile );
  try
    {
    fixedReader->Update();
    }
  catch( itk::ExceptionObject & error )
    {
    std::cerr << "Error when reading meshes: " << error << std::endl;
    return EXIT_FAILURE;
    }
  MeshType::Pointer fixedMesh = fixedReader->GetOutput();
  PointSetType::Pointer fixedPointSet = PointSetType::New();
  fixedPointSet->SetPoints( fixedMesh->GetPoints() );
  fixedPointSet->SetPointData( fixedMesh->GetPointData() );

  MeshReaderType::Pointer movingReader = MeshReaderType::New();
  movingReader->SetFileName( inputMovingMeshFile );
  try
    {
    movingReader->Update();
    }
  catch( itk::ExceptionObject & error )
    {
    std::cerr << "Error when reading meshes: " << error << std::endl;
    return EXIT_FAILURE;
    }
  MeshType::Pointer movingMesh = movingReader->GetOutput();
  PointSetType::Pointer movingPointSet = PointSetType::New();
  movingPointSet->SetPoints( movingMesh->GetPoints() );
  movingPointSet->SetPointData( movingMesh->GetPointData() );

  using ICPPointSetMetricType = itk::EuclideanDistancePointSetToPointSetMetricv4< PointSetType >;
  ICPPointSetMetricType::Pointer icpMetric = ICPPointSetMetricType::New();

  using ExpectationPointSetMetricType = itk::ExpectationBasedPointSetToPointSetMetricv4< PointSetType >;
  ExpectationPointSetMetricType::Pointer expectationMetric = ExpectationPointSetMetricType::New();

  using JHCTPointSetMetricType = itk::JensenHavrdaCharvatTsallisPointSetToPointSetMetricv4< PointSetType >;
  JHCTPointSetMetricType::Pointer jhctMetric = JHCTPointSetMetricType::New();

  using AffineTransformType = itk::AffineTransform<double, Dimension>;
  AffineTransformType::Pointer affineTransform = AffineTransformType::New();
  affineTransform->SetIdentity();
  try
    {
    switch( metricId )
      {
    case 0:
      // ICP
      EuclideanDistancePointSetMetricRegistration<AffineTransformType, ICPPointSetMetricType, PointSetType>
      ( numberOfIterations, maximumPhysicalStepSize,
        affineTransform, icpMetric,
        fixedPointSet, movingPointSet );
      break;
    case 1:
      // GMM
      ExpectationBasedPointSetMetricRegistration<AffineTransformType, ExpectationPointSetMetricType, PointSetType>
      ( numberOfIterations, maximumPhysicalStepSize,
        affineTransform, expectationMetric,
        fixedPointSet, movingPointSet,
        pointSetSigma );
      break;
    case 2:
      // JHCT divergence
      JHCTPointSetMetricRegistration<AffineTransformType, JHCTPointSetMetricType, PointSetType >
      ( numberOfIterations, maximumPhysicalStepSize,
        affineTransform, jhctMetric,
        fixedPointSet, movingPointSet, pointSetSigma );
      break;
    default:
      std::cerr << "Unexpected metric id: " << metricId << std::endl;
      return EXIT_FAILURE;
      }
    }
  catch( itk::ExceptionObject & error )
    {
    std::cerr << "Error during registration: " << error << std::endl;
    }

  using TransformWriterType = itk::TransformFileWriterTemplate< double >;
  TransformWriterType::Pointer transformWriter = TransformWriterType::New();
  transformWriter->SetInput( affineTransform );
  transformWriter->SetFileName( outputTransformFile );

  try
    {
    transformWriter->Update();
    }
  catch( itk::ExceptionObject & error )
    {
    std::cerr << "Error when writing output transform: " << error << std::endl;
    return EXIT_FAILURE;
    }

  return EXIT_SUCCESS;
}
