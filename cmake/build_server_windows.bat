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
SET "MSVS_TOOLSET=14.2"
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
CALL %~dp0build_common_functions.bat :GenerateCTestDashboard build_server_windows.cmake ctest_build_steps.cmake %VIAME_SOURCE_DIR%

"%CMAKE_ROOT%\bin\ctest.exe" -S %VIAME_SOURCE_DIR%\cmake\ctest_build_steps.cmake -VV
IF %ERRORLEVEL% NEQ 0 (
    ECHO CTest build failed with error code %ERRORLEVEL%
    EXIT /B %ERRORLEVEL%
)

REM -------------------------------------------------------------------------------------------------------
REM Final Install Generation Hacks Until Handled Better in VIAME CMake
REM -------------------------------------------------------------------------------------------------------

CALL %~dp0build_common_functions.bat :CopySystemDlls "%VIAME_INSTALL_DIR%\bin" "%VDIST_ROOT%" "%ZLIB_ROOT%" "%ZLIB_BUILD_DIR%"

DEL "%VIAME_INSTALL_DIR%\%PYTHON_SUBDIR%\site-packages\torch\lib\cu*"

CALL %~dp0build_common_functions.bat :CopyCuda12Dlls "%CUDA_ROOT%" "%VIAME_INSTALL_DIR%\bin"
COPY "%CUDA_ROOT%\extras\CUPTI\lib64\cupti64_2025.1.0.dll" %VIAME_INSTALL_DIR%\bin

REM -------------------------------------------------------------------------------------------------------
REM Generate Final Zip File
REM -------------------------------------------------------------------------------------------------------

CALL %~dp0build_common_functions.bat :CreateZipPackage "%VIAME_INSTALL_DIR%" "%VIAME_BUILD_DIR%" "%OUTPUT_FILE%" "%ZIP_ROOT%"
