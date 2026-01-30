message( "Running custom fletch install" )

include( ${VIAME_CMAKE_DIR}/common_macros.cmake )

if( WIN32 )

  if( MSVC AND MSVC_VERSION EQUAL 1900 )
    RenameSubstr( ${VIAME_INSTALL_PREFIX}/lib/libboost* vc120 vc140 )
  endif()
 
  if( VIAME_ENABLE_OPENCV )
    CopyFiles( ${VIAME_INSTALL_PREFIX}/x86/*/bin/*.dll ${VIAME_INSTALL_PREFIX}/bin )
    CopyFiles( ${VIAME_INSTALL_PREFIX}/x86/*/lib/*.lib ${VIAME_INSTALL_PREFIX}/lib )

    CopyFiles( ${VIAME_INSTALL_PREFIX}/x64/*/bin/*.dll ${VIAME_INSTALL_PREFIX}/bin )
    CopyFiles( ${VIAME_INSTALL_PREFIX}/x64/*/lib/*.lib ${VIAME_INSTALL_PREFIX}/lib )
  endif()
endif()

if( NOT WIN32 AND EXISTS ${VIAME_INSTALL_PREFIX}/lib/libpng.so.16 )
  CreateSymlink( ${VIAME_INSTALL_PREFIX}/lib/libpng.so.16
                 ${VIAME_INSTALL_PREFIX}/lib/libpng16.so )
endif()

# Move any misinstalled python files
if( PYTHON_VERSION_STRING )
  # Sometimes fletch subpackages install python files to incorrect python
  # subdirectories, like lib/site-packages instead of lib/pythonX.Y/site-packages
  set( ROOT_PYTHON_DIR "${VIAME_INSTALL_PREFIX}/lib/${PYTHON_VERSION_STRING}" )
  set( OUTPUT_PYTHON_DIR "${ROOT_PYTHON_DIR}/site-packages/" )

  if( EXISTS ${VIAME_INSTALL_PREFIX}/lib/site-packages )
    set( DIR_TO_MOVE "${VIAME_INSTALL_PREFIX}/lib/site-packages" )
    file( GLOB FILES_TO_MOVE "${DIR_TO_MOVE}/*" )
    file( COPY ${FILES_TO_MOVE} DESTINATION ${OUTPUT_PYTHON_DIR} )
    file( REMOVE_RECURSE ${DIR_TO_MOVE} )
  endif()

  if( EXISTS ${VIAME_INSTALL_PREFIX}/lib/python/site-packages )
    set( DIR_TO_MOVE "${VIAME_INSTALL_PREFIX}/lib/python/site-packages" )
    file( GLOB FILES_TO_MOVE "${DIR_TO_MOVE}/*" )
    file( COPY ${FILES_TO_MOVE} DESTINATION ${OUTPUT_PYTHON_DIR} )
    file( REMOVE_RECURSE ${DIR_TO_MOVE} )
  endif()

  file( GLOB OTHER_PYTHON_DIRS "${ROOT_PYTHON_DIR}.*" )

  foreach( ALT_PYTHON_FOLDER ${OTHER_PYTHON_DIRS} )
    file( GLOB FILES_TO_MOVE "${ALT_PYTHON_FOLDER}/site-packages/*" )
    file( COPY ${FILES_TO_MOVE} DESTINATION ${OUTPUT_PYTHON_DIR} )
    file( REMOVE_RECURSE "${ALT_PYTHON_FOLDER}" )
  endforeach()

  # Install OpenCV Python dist-info so pip thinks opencv-python is already installed.
  # This prevents transitive dependencies (e.g. mmengine pulled in by mmdeploy) from
  # installing a pip opencv-python package that overwrites fletch's cv2 Python wrapper
  # with a version that doesn't match the fletch-built cv2 binary.
  if( VIAME_ENABLE_OPENCV )
    set( PATCH_DIR ${VIAME_CMAKE_DIR}/../packages/patches/fletch )
    file( COPY ${PATCH_DIR}/opencv_python-4.9.0.80.dist-info DESTINATION ${OUTPUT_PYTHON_DIR} )
  endif()

  # Patch cv2/__init__.py to catch AttributeError in addition to ImportError when
  # loading extra submodules. This prevents import failures when the cv2 Python
  # wrapper references features not compiled into the cv2 binary (e.g. GStreamer).
  if( VIAME_ENABLE_OPENCV )
    set( CV2_INIT_FILE "${OUTPUT_PYTHON_DIR}/cv2/__init__.py" )
    if( EXISTS "${CV2_INIT_FILE}" )
      file( READ "${CV2_INIT_FILE}" CV2_INIT_DATA )
      string( FIND "${CV2_INIT_DATA}" "except (ImportError, AttributeError)" _cv2_patched )
      if( _cv2_patched EQUAL -1 )
        string( REPLACE
          "except ImportError as err:"
          "except (ImportError, AttributeError) as err:"
          CV2_INIT_DATA "${CV2_INIT_DATA}" )
        file( WRITE "${CV2_INIT_FILE}" "${CV2_INIT_DATA}" )
        message( STATUS "Patched cv2/__init__.py to handle AttributeError in submodule loading" )
      endif()
    endif()
  endif()

  set( PYTHON_ID "${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_VERSION}" )
  set( ANNOYING_FILE "${OUTPUT_PYTHON_DIR}/cv2/config-${PYTHON_ID}.py" )

  if( EXISTS ${ANNOYING_FILE} )
    file( WRITE "${ANNOYING_FILE}"
"PYTHON_EXTENSIONS_PATHS = [\n\
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'python-${PYTHON_ID}')\n\
] + PYTHON_EXTENSIONS_PATHS" )
  endif()

  # Required for certain versions of pytorch or netharn
  if( UNIX AND VIAME_BUILD_PYTHON_FROM_SOURCE )
    set( LZMA_FILE "${ROOT_PYTHON_DIR}/lzma.py" )

    set( SEARCH_CODE1
"from _lzma import *" )
    set( SEARCH_CODE2
"from _lzma import _encode_filter_properties, _decode_filter_properties" )
    set( REPL_CODE
"try:\n\
    from  _lzma import *\n\
    from  _lzma import _encode_filter_properties, _decode_filter_properties\n\
except ImportError:\n\
    from  backports.lzma import *\n\
    from  backports.lzma import _encode_filter_properties, _decode_filter_properties\n" )

    if( EXISTS ${LZMA_FILE} )
      file( READ "${LZMA_FILE}" LZMA_FILE_DATA )
      string( REPLACE "${SEARCH_CODE1}" "${REPL_CODE}" ADJ_FILE_DATA "${LZMA_FILE_DATA}" )
      string( REPLACE "${SEARCH_CODE2}" "" LZMA_FILE_DATA "${ADJ_FILE_DATA}" )
      file( WRITE "${LZMA_FILE}" "${LZMA_FILE_DATA}")
    endif()
  endif()
endif()

message( "Finished fletch custom install" )
