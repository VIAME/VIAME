#include "itkImageFileReader.h"
#include "itkCoherenceEnhancingDiffusionImageFilter.h"
#include "itkPhaseSymmetryImageFilter.h"
#include "itkImageFileWriter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkBinShrinkImageFilter.h"
#include "itkFFTPadImageFilter.h"
#include "itkImage.h"
#include "itkConstantPadImageFilter.h"
#include "itkMaskImageFilter.h"
#include "itkMesh.h"
#include "itkPointSet.h"
#include "itkBinaryMaskToNarrowBandPointSetFilter.h"
#include "itkMeshFileWriter.h"
#include "itkChangeInformationImageFilter.h"
#include "itkJensenHavrdaCharvatTsallisPointSetToPointSetMetricv4.h"
#include "itkGradientDescentOptimizerv4.h"
#include "itkRegistrationParameterScalesFromPhysicalShift.h"
#include "itkAffineTransform.h"
#include "itkCommand.h"
#include "itkTransformFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkImageFileWriter.h"

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

template< typename TPointSet >
typename TPointSet::Pointer
PhaseSymmetryPointSet( const itk::Image< float, 2 >& input, bool isThermal )
{
  constexpr unsigned int Dimension = 2;
  using PixelType = float;
  using ImageType = itk::Image< PixelType, Dimension >;

  using ShrinkerType = itk::BinShrinkImageFilter< ImageType, ImageType >;
  ShrinkerType::Pointer shrinker = ShrinkerType::New();
  shrinker->SetInput( input );
  using ShrinkFactorsType = ShrinkerType::ShrinkFactorsType;
  ShrinkFactorsType shrinkFactors;
  shrinkFactors.Fill( 10 );
  shrinker->SetShrinkFactors( shrinkFactors );

  // Smoothing / noise reduction
  using SmootherType = itk::CoherenceEnhancingDiffusionImageFilter< ImageType >;
  SmootherType::Pointer smoother = SmootherType::New();
  if( isThermal )
    {
    smoother->SetInput( reader->GetOutput() );
    }
  else
    {
    smoother->SetInput( shrinker->GetOutput() );
    }
  smoother->SetEnhancement( SmootherType::cEED );
  if( isThermal )
    {
    smoother->SetDiffusionTime( 1.0 );
    }
  else
    {
    smoother->SetDiffusionTime( 3.0 );
    }

  using FFTPadFilterType = itk::FFTPadImageFilter< ImageType >;
  FFTPadFilterType::Pointer fftPadFilter = FFTPadFilterType::New();
  fftPadFilter->SetInput( smoother->GetOutput() );
  fftPadFilter->Update();

  ImageType::Pointer padded = fftPadFilter->GetOutput();
  padded->DisconnectPipeline();
  ImageType::RegionType paddedRegion( padded->GetBufferedRegion() );
  paddedRegion.SetIndex( 0, 0 );
  paddedRegion.SetIndex( 1, 0 );
  padded->SetRegions( paddedRegion );

  using PhaseSymmetryFilterType = itk::PhaseSymmetryImageFilter< ImageType, ImageType >;
  PhaseSymmetryFilterType::Pointer phaseSymmetryFilter = PhaseSymmetryFilterType::New();
  phaseSymmetryFilter->SetInput( padded );
  phaseSymmetryFilter->SetSigma( 0.25 );
  phaseSymmetryFilter->SetPolarity( 0 );
  if( isThermal )
    {
    phaseSymmetryFilter->SetNoiseThreshold( 15.0 );
    }
  else
    {
    phaseSymmetryFilter->SetNoiseThreshold( 40.0 );
    }
  using MatrixType = PhaseSymmetryFilterType::MatrixType;
  MatrixType wavelengths( 6, Dimension );
  for( unsigned int dim = 0; dim < Dimension; ++dim )
    {
    wavelengths(0, dim) = 2.0;
    wavelengths(1, dim) = 4.0;
    wavelengths(2, dim) = 6.0;
    wavelengths(3, dim) = 8.0;
    wavelengths(4, dim) = 12.0;
    wavelengths(5, dim) = 16.0;
    }
  phaseSymmetryFilter->SetWavelengths( wavelengths );
  smoother->Update();
  phaseSymmetryFilter->Initialize();

  using MaskImageType = itk::Image< unsigned char, Dimension >;
  using ThresholderType = itk::BinaryThresholdImageFilter< ImageType, MaskImageType >;
  typename ThresholderType::Pointer thresholder = ThresholderType::New();
  thresholder->SetInput( phaseSymmetryFilter->GetOutput() );
  thresholder->SetLowerThreshold( 0.01 );
  thresholder->Update();

  // We will need to account for this in the final transform
  using ChangeSpacingFilterType = itk::ChangeInformationImageFilter< MaskImageType >;
  typename ChangeSpacingFilterType::Pointer changeSpacing  = ChangeSpacingFilterType::New();
  changeSpacing->SetInput( thresholder->GetOutput() );
  typename MaskImageType::SpacingType forcedSpacing;
  forcedSpacing.Fill( 1.0 );
  changeSpacing->SetOutputSpacing( forcedSpacing );
  changeSpacing->SetChangeSpacing( true );
  MaskImageType::DirectionType direction;
  direction.SetIdentity();
  changeSpacing->SetOutputDirection( direction );
  changeSpacing->SetChangeDirection( true );
  MaskImageType::PointType origin;
  origin.Fill( 0.0 );
  changeSpacing->SetOutputOrigin( origin );
  changeSpacing->SetChangeOrigin( true );
  changeSpacing->UpdateOutputInformation();

  typename MaskImageType::Pointer inputBinaryMask = changeSpacing->GetOutput();
  using RegionType = typename MaskImageType::RegionType;
  const RegionType inputRegion = inputBinaryMask->GetLargestPossibleRegion();
  typename MaskImageType::SizeType trimSize;
  trimSize.Fill( 10 );
  RegionType regionToKeep( inputRegion );
  regionToKeep.ShrinkByRadius( trimSize );

  typename MaskImageType::Pointer edgeSuppressedBinaryMaskInput = MaskImageType::New();
  edgeSuppressedBinaryMaskInput->SetRegions( regionToKeep );
  edgeSuppressedBinaryMaskInput->Allocate();
  edgeSuppressedBinaryMaskInput->FillBuffer( 1 );
  edgeSuppressedBinaryMaskInput->SetSpacing( inputBinaryMask->GetSpacing() );

  using PadFilterType = itk::ConstantPadImageFilter< MaskImageType, MaskImageType >;
  typename PadFilterType::Pointer padFilter = PadFilterType::New();
  padFilter->SetInput( edgeSuppressedBinaryMaskInput );
  padFilter->SetPadBound( trimSize );

  using MaskFilterType = itk::MaskImageFilter< MaskImageType, MaskImageType, MaskImageType >;
  typename MaskFilterType::Pointer edgeSuppressMaskFilter = MaskFilterType::New();
  edgeSuppressMaskFilter->SetInput( changeSpacing->GetOutput() );
  edgeSuppressMaskFilter->SetMaskImage( padFilter->GetOutput() );
  edgeSuppressMaskFilter->SetMaskingValue( 0 );
  edgeSuppressMaskFilter->SetOutsideValue( 0 );

  using PointSetType = TPointSet;

  using MaskToPointSetFilterType = itk::BinaryMaskToNarrowBandPointSetFilter< MaskImageType, PointSetType >;
  typename MaskToPointSetFilterType::Pointer maskToPointSetFilter = MaskToPointSetFilterType::New();
  maskToPointSetFilter->SetInput( edgeSuppressMaskFilter->GetOutput() );
  constexpr float bandwidth = 0.8f;
  maskToPointSetFilter->SetBandWidth( bandwidth );
  maskToPointSetFilter->Update();

  return maskToPointSetFilter->GetOutput();
}

template< typename InputImageType >
void PerformRegistration(
  const InputImageType& inputOpticalImage,
  const InputImageType& inputThermalImage,
  const outputTransformation,
  const bool ProduceWarped = false,
  const outputWarpedImage = () )
{
  constexpr unsigned int Dimension = 2;
  using PixelType = unsigned char;
  using ImageType = itk::Image< PixelType, Dimension >;
  using PointSetType = itk::PointSet< float, Dimension >;

  PointSetType::Pointer thermalPhaseSymmetryPointSet =
    PhaseSymmetryPointSet< PointSetType >( inputThermalImage, true );
  PointSetType::Pointer opticalPhaseSymmetryPointSet =
    PhaseSymmetryPointSet< PointSetType >( inputOpticalImage, false );

  using JHCTPointSetMetricType = itk::JensenHavrdaCharvatTsallisPointSetToPointSetMetricv4< PointSetType >;
  JHCTPointSetMetricType::Pointer jhctMetric = JHCTPointSetMetricType::New();
  using AffineTransformType = itk::AffineTransform<double, Dimension>;
  AffineTransformType::Pointer affineTransform = AffineTransformType::New();
  affineTransform->SetIdentity();
  constexpr unsigned int numberOfIterations = 100;
  constexpr double maximumPhysicalStepSize = 2.0;
  constexpr double pointSetSigma = 3.0;

  JHCTPointSetMetricRegistration<AffineTransformType, JHCTPointSetMetricType, PointSetType >
    ( numberOfIterations, maximumPhysicalStepSize,
      affineTransform, jhctMetric,
      thermalPhaseSymmetryPointSet,
      opticalPhaseSymmetryPointSet,
      pointSetSigma );
  
  using TransformWriterType = itk::TransformFileWriterTemplate< double >;
  TransformWriterType::Pointer transformWriter = TransformWriterType::New();
  transformWriter->SetInput( affineTransform );
  transformWriter->SetFileName( outputTransformFile );
  transformWriter->Update();

  using ReadPixelType = unsigned short;
  using ReadImageType = itk::Image< ReadPixelType, Dimension >;
  using ImageReaderType = itk::ImageFileReader< ReadImageType >;
  ImageReaderType::Pointer fixedReader = ImageReaderType::New();
  fixedReader->SetFileName( inputThermalImageFile );
  try
    {
    fixedReader->Update();
    }
  catch( itk::ExceptionObject & error )
    {
    std::cerr << "Error when reading fixed image: " << error << std::endl;
    return EXIT_FAILURE;
    }

  ImageReaderType::Pointer movingReader = ImageReaderType::New();
  movingReader->SetFileName( inputOpticalImageFile );
  try
    {
    movingReader->UpdateOutputInformation();
    }
  catch( itk::ExceptionObject & error )
    {
    std::cerr << "Error when reading moving image: " << error << std::endl;
    return EXIT_FAILURE;
    }

  using ResamplerType = itk::ResampleImageFilter< ReadImageType, ReadImageType >;
  ResamplerType::Pointer resampler = ResamplerType::New();
  resampler->SetInput( fixedReader->GetOutput() );
  // Todo: fix handcoded value
  ReadImageType::SpacingType outputSpacing;
  outputSpacing.Fill( 0.1 );
  resampler->SetOutputSpacing( outputSpacing );
  resampler->SetSize( movingReader->GetOutput()->GetLargestPossibleRegion().GetSize() );
  // mid-intensity for the thermal image
  resampler->SetDefaultPixelValue( 27500 );
  try
    {
    resampler->Update();
    }
  catch( itk::ExceptionObject & error )
    {
    std::cerr << "Error when resampling: " << error << std::endl;
    return EXIT_FAILURE;
    }
  ReadImageType::Pointer resampledFixedImage = resampler->GetOutput();
  resampledFixedImage->DisconnectPipeline();
  // For comparison with the inputs, which do not have spacing encoded in
  // their files
  outputSpacing.Fill( 1.0 );
  resampledFixedImage->SetSpacing( outputSpacing );


  AffineTransformType::InverseTransformBasePointer inverseTransform = affineTransform->GetInverseTransform();
  resampler->SetTransform( inverseTransform );
  try
    {
    resampler->Update();
    }
  catch( itk::ExceptionObject & error )
    {
    std::cerr << "Error when resampling: " << error << std::endl;
    return EXIT_FAILURE;
    }
  ReadImageType::Pointer resampledTransformedFixedImage = resampler->GetOutput();
  resampledTransformedFixedImage->DisconnectPipeline();
  // For comparison with the inputs, which do not have spacing encoded in
  // their files
  outputSpacing.Fill( 1.0 );
  resampledTransformedFixedImage->SetSpacing( outputSpacing );

  using WritePixelType = unsigned char;
  using WriteImageType = itk::Image< WritePixelType, Dimension >;
  using RescalerType = itk::RescaleIntensityImageFilter< ReadImageType, WriteImageType >;
  RescalerType::Pointer rescaler = RescalerType::New();
  rescaler->SetInput( resampledFixedImage );

  using ImageWriterType = itk::ImageFileWriter< WriteImageType >;
  //ImageWriterType::Pointer rescaledWriter = ImageWriterType::New();
  //rescaledWriter->SetFileName( rescaledFixedImageFile );
  //rescaledWriter->SetInput( rescaler->GetOutput() );

  RescalerType::Pointer transformedRescaler = RescalerType::New();
  transformedRescaler->SetInput( resampledTransformedFixedImage );


}

int main(int argc, char * argv[])
{
  if( argc < 4 )
    {
    std::cerr << "Usage: " << argv[0] << " <InputThermalImage> <InputOpticalImage> "
              << "<OutputTransformFile> <OutputTransformedThermalImage>" << std::endl;
    std::cerr << "Example: ./0/CHESS_FL12_C_160421_215351.941_THERM-16BIT.PNG "
              << "./0/CHESS_FL12_C_160421_215351.941_COLOR-8-BIT.JPG "
              << "./0/thermal_to_optical.h5 ./0/thermal_registered.png" << std::endl;

    return EXIT_FAILURE;
    }
  
  constexpr unsigned int Dimension = 2;
  using PixelType = unsigned char;
  using ImageType = itk::Image< PixelType, Dimension >;
  using PointSetType = itk::PointSet< float, Dimension >;

  const char * inputThermalImageFile = argv[1];
  const char * inputOpticalImageFile = argv[2];
  const char * outputTransformFile = argv[3];
  const char * outputTransformedThermalImageFile;

  if (argc > 4 )
    {
    outputTransformedThermalImageFile = argv[4];
    }

  using ReaderType = itk::ImageFileReader< PointSetType >;
  ReaderType::Pointer opticalReader = ReaderType::New();
  ReaderType::Pointer thermalReader = ReaderType::New();

  opticalReader->SetFileName( inputOpticalImageFile );
  thermalReader->SetFileName( inputThermalImageFile );

  PerformRegistration( opticalReader->GetOutput(),
                       thermalReader->GetOutput(),
                       ( argc > 4 ),
                       outputTransform,
                       outputWarpedImage );

  ImageWriterType::Pointer transformedRescaledWriter = ImageWriterType::New();
  transformedRescaledWriter->SetInput( transformedRescaler->GetOutput() );
  transformedRescaledWriter->SetFileName( outputTransformedThermalImageFile );

  try
    {
    //rescaledWriter->Update();
    transformedRescaledWriter->Update();
    }
  catch( itk::ExceptionObject & error )
    {
    std::cerr << "Error when writing: " << error << std::endl;
    return EXIT_FAILURE;
    }

  return EXIT_SUCCESS;
}
