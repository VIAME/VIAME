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

REM Location of required libraries installed on the system
SET "INSTALL_DIR_CMAKE=C:\Program Files\CMake"
SET "INSTALL_DIR_GIT=C:\Program Files\Git"
SET "INSTALL_DIR_ZIP=C:\Program Files\7-Zip"
SET "INSTALL_DIR_ZLIB=C:\Program Files\ZLib"
SET "INSTALL_DIR_NODEJS=C:\Program Files\nodejs"
SET "INSTALL_DIR_VDIST=C:\Program Files (x86)\Microsoft Visual Studio"
SET "INSTALL_DIR_VDIST=%INSTALL_DIR_VDIST%\2026\Community\VC\Redist\MSVC"
SET "INSTALL_DIR_VDIST=%INSTALL_DIR_VDIST%\14.40.33807\x64\Microsoft.VC144.OpenMP"

SET "WIN_ROOT=C:\Windows"
SET "WIN32_ROOT=%WIN_ROOT%\System32"
SET "WIN64_ROOT=%WIN_ROOT%\SysWOW64"

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
SET "PATH=%INSTALL_DIR_NODEJS%;%INSTALL_DIR_GIT%\cmd;%INSTALL_DIR_CMAKE%\bin;%PATH%"
SET "PYTHONPATH=%VIAME_INSTALL_DIR%\%PYTHON_SUBDIR%"
SET "PYTHONPATH=%PYTHONPATH%;%VIAME_INSTALL_DIR%\%PYTHON_SUBDIR%\site-packages"

REM --------------------------------------------------------------------------
REM Check Build Dependencies
REM --------------------------------------------------------------------------

CALL %~dp0build_common_functions.bat ^
    :CheckBuildDependencies ^
    "%INSTALL_DIR_CMAKE%" "%INSTALL_DIR_GIT%" ^
    "%INSTALL_DIR_ZIP%" "%INSTALL_DIR_ZLIB%" "SKIP"
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

git config --global core.longpaths true
git submodule update --init --recursive

REM Generate CTest dashboard file
CALL %~dp0build_common_functions.bat ^
    :GenerateCTestDashboard ^
    build_server_windows_cpu.cmake ^
    ctest_build_steps.cmake %VIAME_SOURCE_DIR%

"%INSTALL_DIR_CMAKE%\bin\ctest.exe" ^
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
    "%INSTALL_DIR_VDIST%" "%INSTALL_DIR_ZLIB%" "%ZLIB_BUILD_DIR%"

REM --------------------------------------------------------------------------
REM Generate Final Zip File
REM --------------------------------------------------------------------------

CALL %~dp0build_common_functions.bat ^
    :CreateZipPackage "%VIAME_INSTALL_DIR%" ^
    "%VIAME_BUILD_DIR%" "%OUTPUT_FILE%" "%INSTALL_DIR_ZIP%"
