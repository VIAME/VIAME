#include "itkImageFileReader.h"
#include "itkCoherenceEnhancingDiffusionImageFilter.h"
#include "itkPhaseSymmetryImageFilter.h"
#include "itkImageFileWriter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkBinShrinkImageFilter.h"
#include "itkFFTPadImageFilter.h"

int main(int argc, char * argv[])
{
  if( argc < 4 )
    {
    std::cerr << "Usage: " << argv[0] << " <InputImage> <IsThermal> <OutputImage>" << std::endl;
    std::cerr << "Example: ./0_PhaseSymmetry 0/CHESS_FL12_C_160421_215351.941_THERM-16BIT.PNG 1 ./0/thermal_phase_symmetry.png" << std::endl;
    return EXIT_FAILURE;
    }
  const char * inputImageFile = argv[1];
  bool isThermal = static_cast< bool >( atoi( argv[2] ) );
  const char * outputImageFile = argv[3];

  constexpr unsigned int Dimension = 2;
  using PixelType = float;
  using ImageType = itk::Image< PixelType, Dimension >;

  using ReaderType = itk::ImageFileReader< ImageType >;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName( inputImageFile );

  using ShrinkerType = itk::BinShrinkImageFilter< ImageType, ImageType >;
  ShrinkerType::Pointer shrinker = ShrinkerType::New();
  shrinker->SetInput( reader->GetOutput() );
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
  try
    {
    fftPadFilter->Update();
    }
  catch( itk::ExceptionObject & error )
    {
    std::cerr << "Error: " << error << std::endl;
    return EXIT_FAILURE;
    }

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
  try
    {
    smoother->Update();
    phaseSymmetryFilter->Initialize();
    }
  catch( itk::ExceptionObject & error )
    {
    std::cerr << "Error: " << error << std::endl;
    return EXIT_FAILURE;
    }

  using MaskImageType = itk::Image< unsigned char, Dimension >;
  using ThresholderType = itk::BinaryThresholdImageFilter< ImageType, MaskImageType >;
  ThresholderType::Pointer thresholder = ThresholderType::New();
  thresholder->SetInput( phaseSymmetryFilter->GetOutput() );
  thresholder->SetLowerThreshold( 0.01 );

  using WriterType = itk::ImageFileWriter< MaskImageType >;
  WriterType::Pointer writer = WriterType::New();
  writer->SetInput( thresholder->GetOutput() );
  writer->SetFileName( outputImageFile );

  try
    {
    writer->Update();
    }
  catch( itk::ExceptionObject & error )
    {
    std::cerr << "Error: " << error << std::endl;
    return EXIT_FAILURE;
    }

  return EXIT_SUCCESS;
}
