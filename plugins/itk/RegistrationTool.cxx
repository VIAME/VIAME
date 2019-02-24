#include "RegisterOpticalAndThermal.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkTransformFileWriter.h"

using namespace viame::itk;

int main( int argc, char * argv[] )
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
  
  const char* inputThermalImageFile = argv[1];
  const char* inputOpticalImageFile = argv[2];
  const char* outputTransformFile = argv[3];
  const char* outputTransformedThermalImageFile;

  using OpticalReaderType = itk::ImageFileReader< OpticalImageType >;
  using ThermalReaderType = itk::ImageFileReader< ThermalImageType >;

  OpticalReaderType::Pointer opticalReader = OpticalReaderType::New();
  ThermalReaderType::Pointer thermalReader = ThermalReaderType::New();

  opticalReader->SetFileName( inputOpticalImageFile );
  thermalReader->SetFileName( inputThermalImageFile );

  AffineTransformType::Pointer transformation;

  if( !PerformRegistration( *opticalReader->GetOutput(),
                            *thermalReader->GetOutput(),
                            transformation ) )
    {
    std::cerr << "Error registering images" << std::endl;
    return EXIT_FAILURE;
    }

  using TransformWriterType = itk::TransformFileWriterTemplate< double >;
  TransformWriterType::Pointer transformWriter = TransformWriterType::New();
  transformWriter->SetInput( transformation );
  transformWriter->SetFileName( outputTransformFile );
  transformWriter->Update();

  if( argc > 4 )
    {
    outputTransformedThermalImageFile = argv[4];

    using WarpedWriterType = itk::ImageFileWriter< WarpedImageType >;

    WarpedImageType::Pointer warpedImage;

    if( !WarpImage( *opticalReader->GetOutput(),
                    *thermalReader->GetOutput(),
                    *transformation,
                    warpedImage ) )
      {
      std::cerr << "Error warping image" << std::endl;
      return EXIT_FAILURE;
      }

    WarpedWriterType::Pointer warpedWriter = WarpedWriterType::New();
    warpedWriter->SetInput( warpedImage );
    warpedWriter->SetFileName( outputTransformedThermalImageFile );

    try
      {
      warpedWriter->Update();
      }
    catch( itk::ExceptionObject& error )
      {
      std::cerr << "Error when writing: " << error << std::endl;
      return EXIT_FAILURE;
      }
    }

  return EXIT_SUCCESS;
}

