
set( VIAME_BLACKLISTED_BINARIES
  CartConvert
  cjpeg
  classification
  ConicProj
  convert_cifar_data
  convert_imageset
  convert_mnist_data
  convert_mnist_siamese_data
  fax2ps
  fax2tiff
  finetune_net
  gdal_rasterize
  GeoConvert
  Geod
  GeodesicProj
  GeodSolve
  GeoidEval
  gflags_completions.sh
  gif2h5
  gif2tiff
  Gravity
  h5debug
  h5import
  h5jam
  h5ls
  h5mkgrp
  h5repack
  h5repart
  h5unjam
  jpegtran
  lconvert
  libpng-config
  linguist
  lrelease
  lupdate
  makegeo
  MagneticField
  net_speed_benchmark
  opencv_createsamples
  opencv_haartraining
  opencv_performance
  opencv_traincascade
  opencv_visualisation
  opj_compress
  pal2rgb
  Planimeter
  pluginopedia
  png-config
  ppm2tiff
  qcollectiongenerator
  qdoc3
  qhelpconverter
  qhelpgenerator
  qmlplugindump
  qmlviewer
  qt3to4
  qttracereplay
  ras2tiff
  raw2tiff
  rcc
  rdjpgcom
  rgb2ycbcr
  test_net
  thumbnail
  tiff2bw
  tiff2pdf
  tiff2ps
  tiff2rgba
  tiffcmp
  tiffcp
  tiffcrop
  tiffdither
  tiffdump
  tiffgt
  tiffinfo
  tiffmedian
  tiffset
  tiffsplit
  tjbench
  train_net
  TransverseMercatorProj
  upgrade_net_proto_binary
  upgrade_net_proto_text
  upgrade_solver_proto_text
  vtkEncodeString-6.2
  vtkHashSource-6.2
  vtkParseJava-6.2
  vtkParseOGLExt-6.2
  vtkpython
  vtkWrapHierarchy-6.2
  vtkWrapJava-6.2
  vtkWrapPython-6.2
  vtkWrapPythonInit-6.2
  vtkWrapTcl-6.2
  vtkWrapTclInit-6.2
  wrjpgcom
  xmlpatterns
  xmlpatternsvalidator
  )

foreach( binary_file ${VIAME_BLACKLISTED_BINARIES} )
  if( EXISTS ${VIAME_INSTALL_PREFIX}/bin/${binary_file} )
    file( REMOVE ${VIAME_INSTALL_PREFIX}/bin/${binary_file} )
  endif()
  if( EXISTS ${VIAME_INSTALL_PREFIX}/bin/${binary_file}.exe )
    file( REMOVE ${VIAME_INSTALL_PREFIX}/bin/${binary_file}.exe )
  endif()
endforeach()

# Remove pywin32.pth which references directories (win32, win32\lib) that
# don't exist in the VIAME install layout, causing import errors on startup.
# pywin32 is only a transitive dependency and is not required at runtime.
set( PYTHON_SITE_PACKAGES "${VIAME_INSTALL_PREFIX}/lib/python3.10/site-packages" )
if( EXISTS "${PYTHON_SITE_PACKAGES}/pywin32.pth" )
  file( REMOVE "${PYTHON_SITE_PACKAGES}/pywin32.pth" )
endif()
