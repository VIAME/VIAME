#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkConstantPadImageFilter.h"
#include "itkMaskImageFilter.h"
#include "itkMesh.h"
#include "itkBinaryMaskToNarrowBandPointSetFilter.h"
#include "itkMeshFileWriter.h"
#include "itkChangeInformationImageFilter.h"

template< unsigned int VDimension >
int NarrowBandPointSet( int argc, char * argv[] )
{
  constexpr unsigned int Dimension = VDimension;

  if( argc < 3 )
    {
    std::cerr << "Usage: " << argv[0] << " <InputBinaryMask> <OutputMeshPrefix> [BandWidth]" << std::endl;
    std::cerr << "Example: " << argv[0] << " ./0/thermal_phase_symmetry.png ./0/thermal_phase_symmetry" << std::endl;
    return EXIT_FAILURE;
    }
  const char * inputBinaryMaskFile = argv[1];
  std::string outputMeshFile = std::string(argv[2]);
  if( Dimension == 2 )
    {
    // For further processing
    outputMeshFile += ".gii";
    }
  else
    {
    // For visualization in MeshLab
    outputMeshFile += ".off";
    }
  float bandwidth = 0.8f;
  if( argc > 3 )
    {
    bandwidth = atof( argv[3] );
    }

  using BinaryMaskPixelType = unsigned char;

  using BinaryMaskImageType = itk::Image< BinaryMaskPixelType, Dimension >;

  using ReaderType = itk::ImageFileReader< BinaryMaskImageType >;
  typename ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName( inputBinaryMaskFile );

  // We will need to account for this in the final transform
  using ChangeSpacingFilterType = itk::ChangeInformationImageFilter< BinaryMaskImageType >;
  typename ChangeSpacingFilterType::Pointer changeSpacing  = ChangeSpacingFilterType::New();
  changeSpacing->SetInput( reader->GetOutput() );
  typename BinaryMaskImageType::SpacingType forcedSpacing;
  forcedSpacing.Fill( 1.0 );
  changeSpacing->SetOutputSpacing( forcedSpacing );
  changeSpacing->SetChangeSpacing( true );
  try
    {
    changeSpacing->UpdateOutputInformation();
    }
  catch( itk::ExceptionObject & error )
    {
    std::cerr << "Error during reading input image: " << error << std::endl;
    return EXIT_FAILURE;
    }

  typename BinaryMaskImageType::Pointer inputBinaryMask = changeSpacing->GetOutput();
  using RegionType = typename BinaryMaskImageType::RegionType;
  const RegionType inputRegion = inputBinaryMask->GetLargestPossibleRegion();
  typename BinaryMaskImageType::SizeType trimSize;
  trimSize.Fill( 10 );
  RegionType regionToKeep( inputRegion );
  regionToKeep.ShrinkByRadius( trimSize );

  typename BinaryMaskImageType::Pointer edgeSuppressedBinaryMaskInput = BinaryMaskImageType::New();
  edgeSuppressedBinaryMaskInput->SetRegions( regionToKeep );
  edgeSuppressedBinaryMaskInput->Allocate();
  edgeSuppressedBinaryMaskInput->FillBuffer( 1 );
  edgeSuppressedBinaryMaskInput->SetSpacing( inputBinaryMask->GetSpacing() );

  using PadFilterType = itk::ConstantPadImageFilter< BinaryMaskImageType, BinaryMaskImageType >;
  typename PadFilterType::Pointer padFilter = PadFilterType::New();
  padFilter->SetInput( edgeSuppressedBinaryMaskInput );
  padFilter->SetPadBound( trimSize );

  using MaskFilterType = itk::MaskImageFilter< BinaryMaskImageType, BinaryMaskImageType, BinaryMaskImageType >;
  typename MaskFilterType::Pointer edgeSuppressMaskFilter = MaskFilterType::New();
  edgeSuppressMaskFilter->SetInput( changeSpacing->GetOutput() );
  edgeSuppressMaskFilter->SetMaskImage( padFilter->GetOutput() );
  edgeSuppressMaskFilter->SetMaskingValue( 0 );
  edgeSuppressMaskFilter->SetOutsideValue( 0 );

  using MeshType = itk::Mesh< float, Dimension >;

  using MaskToPointSetFilterType = itk::BinaryMaskToNarrowBandPointSetFilter< BinaryMaskImageType, MeshType >;
  typename MaskToPointSetFilterType::Pointer maskToPointSetFilter = MaskToPointSetFilterType::New();
  maskToPointSetFilter->SetInput( edgeSuppressMaskFilter->GetOutput() );
  maskToPointSetFilter->SetBandWidth( bandwidth );

  using MeshWriterType = itk::MeshFileWriter< MeshType >;
  typename MeshWriterType::Pointer meshWriter = MeshWriterType::New();
  meshWriter->SetInput( maskToPointSetFilter->GetOutput() );
  meshWriter->SetFileName( outputMeshFile );

  try
    {
    meshWriter->Update();
    }
  catch( itk::ExceptionObject & error )
    {
    std::cerr << "Error: " << error << std::endl;
    return EXIT_FAILURE;
    }
  return EXIT_SUCCESS;
}

int main(int argc, char * argv[])
{
  if( NarrowBandPointSet<2>( argc, argv ) == EXIT_FAILURE )
    {
    return EXIT_FAILURE;
    }
  return NarrowBandPointSet<3>( argc, argv );
}
