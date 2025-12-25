REM -------------------------------------------------------------------------------------------------------
REM Setup Paths
REM -------------------------------------------------------------------------------------------------------

SET "VIAME_SOURCE_DIR=C:\VIAME-Builds\GPU"

REM Extract version from RELEASE_NOTES.md (first token of first line)
FOR /f "tokens=1 delims= " %%a IN (%VIAME_SOURCE_DIR%\RELEASE_NOTES.md) DO (
    SET "VIAME_VERSION=%%a"
    GOTO :LOOPEXIT
)
:LOOPEXIT

SET "OUTPUT_FILE=VIAME-%VIAME_VERSION%-Windows-64Bit.zip"

REM Make sure to have all of these things installed (and cuDNN in CUDA_ROOT)

SET "CMAKE_ROOT=C:\Program Files\CMake"
SET "GIT_ROOT=C:\Program Files\Git"
SET "ZIP_ROOT=C:\Program Files\7-Zip"
SET "ZLIB_ROOT=C:\Program Files\ZLib"
SET "NVIDIA_ROOT=C:\Program Files (x86)\NVIDIA Corporation"
SET "CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"

SET "MSVS_ROOT=C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional"
SET "MSVS_ARCH=x64"
SET "MSVS_TOOLSET=14.2
SET "MSVS_REDIST_VER=14.29.30133"
SET "WIN_ROOT=C:\Windows"
SET "WIN32_ROOT=%WIN_ROOT%\System32"
SET "WIN64_ROOT=%WIN_ROOT%\SysWOW64"

REM Do not modify the below unless you are changing python versions or have alternatively modified
REM the build and install directories in cmake or the platforms.cmake file

SET "VIAME_BUILD_DIR=%VIAME_SOURCE_DIR%\build"
SET "VIAME_INSTALL_DIR=%VIAME_BUILD_DIR%\install"

SET "PYTHON_SUBDIR=lib\python3.10"
SET "ZLIB_BUILD_DIR=%VIAME_BUILD_DIR%\build\src\fletch-build\build\src\ZLib-build"

SET "PATH=%WIN_ROOT%;%WIN32_ROOT%;%WIN32_ROOT%\Wbem;%WIN32_ROOT%\WindowsPowerShell\v1.0;%WIN32_ROOT%\OpenSSH"
SET "PATH=%CUDA_ROOT%\bin;%CUDA_ROOT%\libnvvp;%NVIDIA_ROOT%\PhysX\Common;%NVIDIA_ROOT%\NVIDIA NvDLISR;%PATH%"
SET "PATH=%GIT_ROOT%\cmd;%CMAKE_ROOT%\bin;%PATH%"
SET "PYTHONPATH=%VIAME_INSTALL_DIR%\%PYTHON_SUBDIR%;%VIAME_INSTALL_DIR%\%PYTHON_SUBDIR%\site-packages"

SET "VDIST_VER_STR=%MSVS_TOOLSET:.=%"
SET "VDIST_ROOT=%MSVS_ROOT%\VC\Redist\MSVC\%MSVS_REDIST_VER%\%MSVS_ARCH%\Microsoft.VC%MSVS_TOOLSET%.OpenMP"

REM -------------------------------------------------------------------------------------------------------
REM Perform Actual Build
REM -------------------------------------------------------------------------------------------------------

REM This build proceedure currently requires making TMP directories at C:\tmp to get around paths
REM which sometimes become too long for windows.

IF "%1"=="true" (
  ECHO "Not erasing build folder"
) ELSE (
  IF EXIST build rmdir /s /q build

  IF NOT EXIST C:\tmp mkdir C:\tmp
  IF EXIST C:\tmp\fl1 rmdir /s /q C:\tmp\fl1
  IF EXIST C:\tmp\kv1 rmdir /s /q C:\tmp\kv1
  IF EXIST C:\tmp\vm1 rmdir /s /q C:\tmp\vm1
)

git config --system core.longpaths true
git submodule update --init --recursive

REM Generate CTest dashboard file
CALL :GenerateCTestDashboard build_server_windows.cmake ctest_dashboard.cmake

"%CMAKE_ROOT%\bin\ctest.exe" -S %VIAME_SOURCE_DIR%\cmake\ctest_dashboard.cmake -VV

REM -------------------------------------------------------------------------------------------------------
REM Final Install Generation Hacks Until Handled Better in VIAME CMake
REM -------------------------------------------------------------------------------------------------------

CALL :CopySystemDlls "%VIAME_INSTALL_DIR%\bin" "%VDIST_ROOT%" "%ZLIB_ROOT%" "%ZLIB_BUILD_DIR%"

DEL "%VIAME_INSTALL_DIR%\%PYTHON_SUBDIR%\site-packages\torch\lib\cu*"

CALL :CopyCuda12Dlls "%CUDA_ROOT%" "%VIAME_INSTALL_DIR%\bin"
COPY "%CUDA_ROOT%\extras\CUPTI\lib64\cupti64_2025.1.0.dll" %VIAME_INSTALL_DIR%\bin

REM -------------------------------------------------------------------------------------------------------
REM Generate Final Zip File
REM -------------------------------------------------------------------------------------------------------

CALL :CreateZipPackage "%VIAME_INSTALL_DIR%" "%VIAME_BUILD_DIR%" "%OUTPUT_FILE%" "%ZIP_ROOT%"

GOTO :EOF

REM ==============================================================================
REM Subroutines
REM ==============================================================================

:GenerateCTestDashboard
(
    ECHO # Auto-generated CTest dashboard file
    ECHO include^(%~1^)
    ECHO ctest_start^(${CTEST_BUILD_MODEL}^)
    ECHO ctest_configure^(BUILD ${CTEST_BINARY_DIRECTORY} SOURCE ${CTEST_SOURCE_DIRECTORY} OPTIONS "${OPTIONS}"^)
    ECHO ctest_build^(^)
    ECHO ctest_submit^(^)
) > %VIAME_SOURCE_DIR%\cmake\%~2
GOTO :EOF

:CopySystemDlls
COPY "%WINDIR%\System32\msvcr100.dll" "%~1" 2>NUL
COPY "%WINDIR%\System32\vcruntime140_1.dll" "%~1" 2>NUL
COPY "%~2\vcomp140.dll" "%~1" 2>NUL
COPY "%WINDIR%\SysWOW64\msvcr120.dll" "%~1" 2>NUL
COPY "%~3\dll_x64\zlibwapi.dll" "%~1" 2>NUL
IF NOT "%~4"=="" COPY "%~4\Release\zlib1.dll" "%~1" 2>NUL
GOTO :EOF

:CopyCuda12Dlls
COPY "%~1\bin\cublas64_12.dll" "%~2" 2>NUL
COPY "%~1\bin\cublasLt64_12.dll" "%~2" 2>NUL
COPY "%~1\bin\cudart64_12.dll" "%~2" 2>NUL
COPY "%~1\bin\cudnn_adv64_9.dll" "%~2" 2>NUL
COPY "%~1\bin\cudnn_cnn64_9.dll" "%~2" 2>NUL
COPY "%~1\bin\cudnn_engines_precompiled64_9.dll" "%~2" 2>NUL
COPY "%~1\bin\cudnn_engines_runtime_compiled64_9.dll" "%~2" 2>NUL
COPY "%~1\bin\cudnn_graph64_9.dll" "%~2" 2>NUL
COPY "%~1\bin\cudnn_heuristic64_9.dll" "%~2" 2>NUL
COPY "%~1\bin\cudnn_ops64_9.dll" "%~2" 2>NUL
COPY "%~1\bin\cudnn64_9.dll" "%~2" 2>NUL
COPY "%~1\bin\cufft64_11.dll" "%~2" 2>NUL
COPY "%~1\bin\cufftw64_11.dll" "%~2" 2>NUL
COPY "%~1\bin\curand64_10.dll" "%~2" 2>NUL
COPY "%~1\bin\cusolver64_11.dll" "%~2" 2>NUL
COPY "%~1\bin\cusolverMg64_11.dll" "%~2" 2>NUL
COPY "%~1\bin\cusparse64_12.dll" "%~2" 2>NUL
GOTO :EOF

:CreateZipPackage
MOVE "%~1" "%~2\VIAME"
"%~4\7z.exe" a "%~2\%~3" "%~2\VIAME"
MOVE "%~2\VIAME" "%~1"
GOTO :EOF
