@ECHO OFF
SETLOCAL EnableDelayedExpansion

REM --------------------------------------------------------------------------
REM Setup Paths
REM --------------------------------------------------------------------------

SET "VIAME_SOURCE_DIR=C:\VIAME-Builds\CPU"

REM Extract version from RELEASE_NOTES.md (first token of first line)
FOR /f "tokens=1 delims= " %%a IN (%VIAME_SOURCE_DIR%\RELEASE_NOTES.md) DO (
    SET "VIAME_VERSION=%%a"
    GOTO :LOOPEXIT
)
:LOOPEXIT

SET "OUTPUT_FILE=VIAME-CPU-%VIAME_VERSION%-Windows-64Bit.zip"

REM Make sure to have all of these things installed

SET "CMAKE_ROOT=C:\Program Files\CMake"
SET "GIT_ROOT=C:\Program Files\Git"
SET "ZIP_ROOT=C:\Program Files\7-Zip"
SET "ZLIB_ROOT=C:\Program Files\ZLib"
SET "NODEJS_ROOT=C:\Program Files\nodejs"

SET "WIN_ROOT=C:\Windows"
SET "WIN32_ROOT=%WIN_ROOT%\System32"
SET "WIN64_ROOT=%WIN_ROOT%\SysWOW64"
SET "VDIST_ROOT=C:\Program Files (x86)\Microsoft Visual Studio"
SET "VDIST_ROOT=%VDIST_ROOT%\2026\Community\VC\Redist\MSVC"
SET "VDIST_ROOT=%VDIST_ROOT%\14.40.33807\x64\Microsoft.VC144.OpenMP"

REM Do not modify the below unless you are changing python
REM versions or have alternatively modified the build and
REM install directories in cmake or the platforms.cmake file

SET "VIAME_BUILD_DIR=%VIAME_SOURCE_DIR%\build"
SET "VIAME_INSTALL_DIR=%VIAME_BUILD_DIR%\install"

SET "PYTHON_SUBDIR=lib\python3.10"
SET "ZLIB_BUILD_DIR=%VIAME_BUILD_DIR%\build\src\fletch-build"
SET "ZLIB_BUILD_DIR=%ZLIB_BUILD_DIR%\build\src\ZLib-build"

SET "PATH=%WIN_ROOT%;%WIN32_ROOT%"
SET "PATH=%PATH%;%WIN32_ROOT%\Wbem"
SET "PATH=%PATH%;%WIN32_ROOT%\WindowsPowerShell\v1.0"
SET "PATH=%PATH%;%WIN32_ROOT%\OpenSSH"
SET "PATH=%NODEJS_ROOT%;%GIT_ROOT%\cmd;%CMAKE_ROOT%\bin;%PATH%"
SET "PYTHONPATH=%VIAME_INSTALL_DIR%\%PYTHON_SUBDIR%"
SET "PYTHONPATH=%PYTHONPATH%;%VIAME_INSTALL_DIR%\%PYTHON_SUBDIR%\site-packages"

REM --------------------------------------------------------------------------
REM Check Build Dependencies
REM --------------------------------------------------------------------------

CALL %~dp0build_common_functions.bat ^
    :CheckBuildDependencies ^
    "%CMAKE_ROOT%" "%GIT_ROOT%" ^
    "%ZIP_ROOT%" "%ZLIB_ROOT%" "SKIP"
IF ERRORLEVEL 1 EXIT /B 1

REM --------------------------------------------------------------------------
REM Perform Actual Build
REM --------------------------------------------------------------------------

REM This build proceedure currently requires making TMP
REM directories at C:\tmp to get around paths which
REM sometimes become too long for windows.

IF EXIST build rmdir /s /q build

IF NOT EXIST C:\tmp mkdir C:\tmp
IF EXIST C:\tmp\kv2 rmdir /s /q C:\tmp\kv2
IF EXIST C:\tmp\vm2 rmdir /s /q C:\tmp\vm2

git config --system core.longpaths true
git submodule update --init --recursive

REM Generate CTest dashboard file
CALL %~dp0build_common_functions.bat ^
    :GenerateCTestDashboard ^
    build_server_windows_cpu.cmake ^
    ctest_build_steps.cmake %VIAME_SOURCE_DIR%

"%CMAKE_ROOT%\bin\ctest.exe" ^
    -S %VIAME_SOURCE_DIR%\cmake\ctest_build_steps.cmake -VV
IF %ERRORLEVEL% NEQ 0 (
    ECHO CTest build failed with error code %ERRORLEVEL%
    EXIT /B %ERRORLEVEL%
)

REM --------------------------------------------------------------------------
REM Final Install Generation Hacks
REM Until Handled Better in VIAME CMake
REM --------------------------------------------------------------------------

CALL %~dp0build_common_functions.bat ^
    :CopySystemDlls "%VIAME_INSTALL_DIR%\bin" ^
    "%VDIST_ROOT%" "%ZLIB_ROOT%" "%ZLIB_BUILD_DIR%"

REM --------------------------------------------------------------------------
REM Generate Final Zip File
REM --------------------------------------------------------------------------

CALL %~dp0build_common_functions.bat ^
    :CreateZipPackage "%VIAME_INSTALL_DIR%" ^
    "%VIAME_BUILD_DIR%" "%OUTPUT_FILE%" "%ZIP_ROOT%"
