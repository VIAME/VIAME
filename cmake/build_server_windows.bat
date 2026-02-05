@ECHO OFF
SETLOCAL EnableDelayedExpansion

REM --------------------------------------------------------------------------
REM Setup Paths
REM --------------------------------------------------------------------------

SET "VIAME_SOURCE_DIR=C:\VIAME-Builds\GPU"

REM Extract version from RELEASE_NOTES.md (first token of first line)
FOR /f "tokens=1 delims= " %%a IN (%VIAME_SOURCE_DIR%\RELEASE_NOTES.md) DO (
    SET "VIAME_VERSION=%%a"
    GOTO :LOOPEXIT
)
:LOOPEXIT

SET "OUTPUT_FILE=VIAME-%VIAME_VERSION%-Windows-64Bit.zip"

REM Location of required libraries installed on the system
SET "INSTALL_DIR_CMAKE=C:\Program Files\CMake"
SET "INSTALL_DIR_GIT=C:\Program Files\Git"
SET "INSTALL_DIR_ZIP=C:\Program Files\7-Zip"
SET "INSTALL_DIR_ZLIB=C:\Program Files\ZLib"
SET "INSTALL_DIR_NODEJS=C:\Program Files\nodejs"
SET "INSTALL_DIR_NVIDIA=C:\Program Files (x86)\NVIDIA Corporation"
SET "INSTALL_DIR_CUDA=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
SET "INSTALL_DIR_MSVS=C:\Program Files\Microsoft Visual Studio\18\Community"

SET "MSVS_ARCH=x64"
SET "MSVS_TOOLSET=14.5"
SET "MSVS_REDIST_VER=14.50.35710"
SET "WIN_ROOT=C:\Windows"
SET "WIN32_ROOT=%WIN_ROOT%\System32"
SET "WIN64_ROOT=%WIN_ROOT%\SysWOW64"

REM Do not modify the below unless you are changing python
REM versions or have alternatively modified the build and
REM install directories in cmake or the platforms.cmake file

SET "VIAME_BUILD_DIR=%VIAME_SOURCE_DIR%\build"
SET "VIAME_INSTALL_DIR=%VIAME_BUILD_DIR%\install"

SET "PYTHON_SUBDIR=lib\python3.10"

SET "PATH=%WIN_ROOT%;%WIN32_ROOT%"
SET "PATH=%PATH%;%WIN32_ROOT%\Wbem"
SET "PATH=%PATH%;%WIN32_ROOT%\WindowsPowerShell\v1.0"
SET "PATH=%PATH%;%WIN32_ROOT%\OpenSSH"
SET "PATH=%INSTALL_DIR_CUDA%\bin;%INSTALL_DIR_CUDA%\libnvvp;%PATH%"
SET "PATH=%INSTALL_DIR_NVIDIA%\PhysX\Common;%PATH%"
SET "PATH=%INSTALL_DIR_NVIDIA%\NVIDIA NvDLISR;%PATH%"
SET "PATH=%APPDATA%\npm;%INSTALL_DIR_NODEJS%;%PATH%"
SET "PATH=%INSTALL_DIR_GIT%\cmd;%INSTALL_DIR_CMAKE%\bin;%PATH%"
SET "PYTHONPATH=%VIAME_INSTALL_DIR%\%PYTHON_SUBDIR%"
SET "PYTHONPATH=%PYTHONPATH%;%VIAME_INSTALL_DIR%\%PYTHON_SUBDIR%\site-packages"

SET "VDIST_VER_STR=%MSVS_TOOLSET:.=%"
SET "INSTALL_DIR_VDIST=%INSTALL_DIR_MSVS%\VC\Redist\MSVC"
SET "INSTALL_DIR_VDIST=%INSTALL_DIR_VDIST%\%MSVS_REDIST_VER%"
SET "INSTALL_DIR_VDIST=%INSTALL_DIR_VDIST%\%MSVS_ARCH%\Microsoft.VC%VDIST_VER_STR%.OpenMP"

REM --------------------------------------------------------------------------
REM Check Build Dependencies
REM --------------------------------------------------------------------------

CALL %~dp0build_common_functions.bat ^
    :CheckBuildDependencies ^
    "%INSTALL_DIR_CMAKE%" "%INSTALL_DIR_GIT%" ^
    "%INSTALL_DIR_ZIP%" "%INSTALL_DIR_ZLIB%" "%INSTALL_DIR_CUDA%"
IF ERRORLEVEL 1 EXIT /B 1

REM --------------------------------------------------------------------------
REM Perform Actual Build
REM --------------------------------------------------------------------------

REM This build proceedure currently requires making TMP
REM directories at C:\tmp to get around paths which
REM sometimes become too long for windows.

IF "%1"=="true" (
  ECHO "Not erasing build folder"
) ELSE (
  IF EXIST build rmdir /s /q build

  IF NOT EXIST C:\tmp mkdir C:\tmp
  IF EXIST C:\tmp\fl1 rmdir /s /q C:\tmp\fl1
  IF EXIST C:\tmp\kv1 rmdir /s /q C:\tmp\kv1
  IF EXIST C:\tmp\vm1 rmdir /s /q C:\tmp\vm1
)

git config --global core.longpaths true
git submodule update --init --recursive

REM Generate CTest dashboard file
CALL %~dp0build_common_functions.bat ^
    :GenerateCTestDashboard ^
    build_server_windows.cmake ^
    ctest_build_steps.cmake %VIAME_SOURCE_DIR%

"%INSTALL_DIR_CMAKE%\bin\ctest.exe" ^
    -S %VIAME_SOURCE_DIR%\cmake\ctest_build_steps.cmake -VV
IF %ERRORLEVEL% NEQ 0 (
    ECHO.
    ECHO ============================================================
    ECHO WARNING: CTest returned error code %ERRORLEVEL%
    ECHO This may be due to compiler warnings/errors detected
    ECHO by CTest launchers.
    ECHO ============================================================
    ECHO.
)

REM Check if build actually completed
IF NOT EXIST "%VIAME_INSTALL_DIR%\setup_viame.bat" (
    ECHO.
    ECHO ==============================================
    ECHO ERROR: Build did not complete successfully
    ECHO setup_viame.bat not found in install directory
    ECHO ==============================================
    EXIT /B 1
)

ECHO.
ECHO ==========================================================
ECHO Build completed, proceeding with final install steps...
ECHO ==========================================================
ECHO.

REM --------------------------------------------------------------------------
REM Read build paths from CMakeCache
REM (single source of truth is the .cmake platform file)
REM --------------------------------------------------------------------------

FOR /f "tokens=2 delims==" %%a IN ('FINDSTR /B "VIAME_BUILD_FLETCH_DIR:" "%VIAME_BUILD_DIR%\CMakeCache.txt"') DO SET "FLETCH_BUILD_DIR=%%a"
SET "ZLIB_BUILD_DIR=%FLETCH_BUILD_DIR%\build\src\ZLib-build"

REM --------------------------------------------------------------------------
REM Final Install Generation Hacks
REM Until Handled Better in VIAME CMake
REM --------------------------------------------------------------------------

CALL %~dp0build_common_functions.bat ^
    :CopySystemDlls "%VIAME_INSTALL_DIR%\bin" ^
    "%INSTALL_DIR_VDIST%" "%INSTALL_DIR_ZLIB%" "%ZLIB_BUILD_DIR%"
CALL %~dp0build_common_functions.bat ^
    :CopyMsysDlls "%FLETCH_BUILD_DIR%" ^
    "%VIAME_INSTALL_DIR%\bin"

IF EXIST "%VIAME_INSTALL_DIR%\%PYTHON_SUBDIR%\site-packages\torch\lib" (
  DEL /Q "%VIAME_INSTALL_DIR%\%PYTHON_SUBDIR%\site-packages\torch\lib\cu*"
)

CALL %~dp0build_common_functions.bat ^
    :CopyCuda12Dlls "%INSTALL_DIR_CUDA%" ^
    "%VIAME_INSTALL_DIR%\bin"

REM --------------------------------------------------------------------------
REM Generate Final Zip File
REM --------------------------------------------------------------------------

CALL %~dp0build_common_functions.bat ^
    :CreateZipPackage "%VIAME_INSTALL_DIR%" ^
    "%VIAME_BUILD_DIR%" "%OUTPUT_FILE%" "%INSTALL_DIR_ZIP%"
IF ERRORLEVEL 1 (
    ECHO.
    ECHO ========================================
    ECHO ERROR: Failed to create zip package
    ECHO ========================================
    EXIT /B 1
)

ECHO.
ECHO ========================================
ECHO Build Completed Successfully
ECHO ========================================
ECHO Output: %VIAME_BUILD_DIR%\%OUTPUT_FILE%
ECHO.
