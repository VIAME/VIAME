#include "itkMeshFileReader.h"
#include "itkMeshFileWriter.h"
#include "itkMesh.h"
#include "itkAffineTransform.h"
#include "itkTransformFileReader.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkResampleImageFilter.h"

template< typename TTransform >
typename TTransform::Pointer
ReadTransform( const char * fileName )
{
  using TransformReaderType = itk::TransformFileReaderTemplate< double >;
  TransformReaderType::Pointer transformReader = TransformReaderType::New();
  transformReader->SetFileName( fileName );

  transformReader->Update();
  typename TTransform::Pointer transform =
    dynamic_cast< TTransform * >( transformReader->GetTransformList()->front().GetPointer() );

  return transform;
}

int main(int argc, char * argv[])
{
  if( argc < 8 )
    {
    std::cerr << "Usage: " << argv[0] << " <FixedToMovingTransform> <FixedPointSet> "
              << "<TransformedFixedPointSet> <FixedImage> <RescaledFixedImage> "
              << "<TransformedRescaledFixedImage> <MovingImage>" << std::endl;

    return EXIT_FAILURE;
    }

  const char * fixedToMovingTransformFile = argv[1];
  const char * fixedPointSetFile = argv[2];
  const char * transformedFixedPointSetFile = argv[3];
  const char * fixedImageFile = argv[4];
  const char * rescaledFixedImageFile = argv[5];
  const char * transformedFixedImageFile = argv[6];
  const char * movingImageFile = argv[7];

  constexpr unsigned int Dimension = 2;
  // Current work around for visualization purposes
  constexpr unsigned int MeshDimension = 3;
  using PointSetType = itk::PointSet<unsigned char, MeshDimension>;
  using MeshType = itk::Mesh<unsigned char, MeshDimension>;

  using MeshReaderType = itk::MeshFileReader< MeshType >;
  MeshReaderType::Pointer meshReader = MeshReaderType::New();
  meshReader->SetFileName( fixedPointSetFile );
  try
    {
    meshReader->Update();
    }
  catch( itk::ExceptionObject & error )
    {
    std::cerr << "Error when reading meshes: " << error << std::endl;
    return EXIT_FAILURE;
    }
  MeshType::Pointer mesh = meshReader->GetOutput();

  using AffineTransformType = itk::AffineTransform<double, Dimension>;
  AffineTransformType::Pointer transform;
  try
    {
    transform = ReadTransform< AffineTransformType >( fixedToMovingTransformFile );
    }
  catch( itk::ExceptionObject & error )
    {
    std::cerr << "Error when reading transform: " << error << std::endl;
    return EXIT_FAILURE;
    }

  using PointType = PointSetType::PointType;
  using TransformPointType = AffineTransformType::InputPointType;
  using PointIdentifierType = PointSetType::PointIdentifier;
  const PointIdentifierType numberOfPoints = mesh->GetNumberOfPoints();
  PointType transformedPoint;
  TransformPointType transformedPoint2D;
  for( PointIdentifierType pointId = 0; pointId < numberOfPoints; ++pointId )
    {
    mesh->GetPoint( pointId, &transformedPoint );
    transformedPoint2D[0] = transformedPoint[0];
    transformedPoint2D[1] = transformedPoint[1];
    transformedPoint2D = transform->TransformPoint( transformedPoint2D );
    transformedPoint[0] = transformedPoint2D[0];
    transformedPoint[1] = transformedPoint2D[1];
    mesh->SetPoint( pointId, transformedPoint );
    }

  using MeshWriterType = itk::MeshFileWriter< MeshType >;
  MeshWriterType::Pointer meshWriter = MeshWriterType::New();
  meshWriter->SetFileName( transformedFixedPointSetFile );
  meshWriter->SetInput( mesh );

  try
    {
    meshWriter->Update();
    }
  catch( itk::ExceptionObject & error )
    {
    std::cerr << "Error when writing mesh: " << error << std::endl;
    return EXIT_FAILURE;
    }

  using ReadPixelType = unsigned short;
  using ReadImageType = itk::Image< ReadPixelType, Dimension >;
  using ImageReaderType = itk::ImageFileReader< ReadImageType >;
  ImageReaderType::Pointer fixedReader = ImageReaderType::New();
  fixedReader->SetFileName( fixedImageFile );
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
  movingReader->SetFileName( movingImageFile );
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


  AffineTransformType::InverseTransformBasePointer inverseTransform = transform->GetInverseTransform();
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
  ImageWriterType::Pointer rescaledWriter = ImageWriterType::New();
  rescaledWriter->SetFileName( rescaledFixedImageFile );
  rescaledWriter->SetInput( rescaler->GetOutput() );

  RescalerType::Pointer transformedRescaler = RescalerType::New();
  transformedRescaler->SetInput( resampledTransformedFixedImage );

  ImageWriterType::Pointer transformedRescaledWriter = ImageWriterType::New();
  transformedRescaledWriter->SetInput( transformedRescaler->GetOutput() );
  transformedRescaledWriter->SetFileName( transformedFixedImageFile );

  try
    {
    rescaledWriter->Update();
    transformedRescaledWriter->Update();
    }
  catch( itk::ExceptionObject & error )
    {
    std::cerr << "Error when writing: " << error << std::endl;
    return EXIT_FAILURE;
    }

  return EXIT_SUCCESS;
}
