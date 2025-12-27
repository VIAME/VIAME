REM -------------------------------------------------------------------------------------------------------
REM VIAME Windows MSI Build Script
REM
REM Builds VIAME in stages to create separate installable packages:
REM   1. Base     - fletch + kwiver (CPU only)
REM   2. CUDA     - adds CUDA/cuDNN support
REM   3. PyTorch  - adds PyTorch + pytorch-libs
REM   4. VIVIA    - adds VIVIA interface
REM   5. SEAL     - adds SEAL toolkit
REM   6. DIVE     - adds DIVE interface
REM
REM Uses VIAME_MSI_STAGE environment variable to control cmake configuration
REM -------------------------------------------------------------------------------------------------------

REM -------------------------------------------------------------------------------------------------------
REM Setup Paths
REM -------------------------------------------------------------------------------------------------------

SET VIAME_SOURCE_DIR=C:\VIAME-Builds\GPU-MSI
SET VIAME_BUILD_DIR=%VIAME_SOURCE_DIR%\build
SET VIAME_INSTALL_DIR=%VIAME_BUILD_DIR%\install

REM Make sure to have all of these things installed (and cuDNN in CUDA)

SET "CMAKE_ROOT=C:\Program Files\CMake"
SET "GIT_ROOT=C:\Program Files\Git"
SET "ZIP_ROOT=C:\Program Files\7-Zip"
SET "ZLIB_ROOT=C:\Program Files\ZLib"
SET "NVIDIA_ROOT=C:\Program Files (x86)\NVIDIA Corporation"
SET "CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"

SET "WIN_ROOT=C:\Windows"
SET "WIN32_ROOT=%WIN_ROOT%\System32"
SET "WIN64_ROOT=%WIN_ROOT%\SysWOW64"

SET "PYTHON_SUBDIR=lib\python3.10"

SET "PATH=%WIN_ROOT%;%WIN32_ROOT%;%WIN32_ROOT%\Wbem;%WIN32_ROOT%\WindowsPowerShell\v1.0;%WIN32_ROOT%\OpenSSH"
SET "PATH=%CUDA_ROOT%\bin;%CUDA_ROOT%\libnvvp;%NVIDIA_ROOT%\PhysX\Common;%NVIDIA_ROOT%\NVIDIA NvDLISR;%PATH%"
SET "PATH=%GIT_ROOT%\cmd;%CMAKE_ROOT%\bin;%PATH%"
SET "PYTHONPATH=%VIAME_INSTALL_DIR%\%PYTHON_SUBDIR%;%VIAME_INSTALL_DIR%\%PYTHON_SUBDIR%\site-packages"

REM -------------------------------------------------------------------------------------------------------
REM Initialize Build Environment
REM -------------------------------------------------------------------------------------------------------

IF EXIST build rmdir /s /q build

IF NOT EXIST C:\tmp mkdir C:\tmp
IF EXIST C:\tmp\fl5 rmdir /s /q C:\tmp\fl5
IF EXIST C:\tmp\kv5 rmdir /s /q C:\tmp\kv5
IF EXIST C:\tmp\vm5 rmdir /s /q C:\tmp\vm5

git config --system core.longpaths true
git submodule update --init --recursive

REM -------------------------------------------------------------------------------------------------------
REM Stage 1 - Base (fletch + kwiver, CPU only)
REM -------------------------------------------------------------------------------------------------------

CALL :BuildStage base "Base (fletch + kwiver)"

CALL %~dp0build_common_functions.bat :CopySystemDlls "%VIAME_INSTALL_DIR%\bin" "%WIN64_ROOT%" "%ZLIB_ROOT%"

CALL :SnapshotFiles files-base.txt
"%ZIP_ROOT%\7z.exe" a -tzip "%VIAME_BUILD_DIR%\VIAME-Base.zip" @files-base.txt

MOVE "%VIAME_INSTALL_DIR%" "%VIAME_BUILD_DIR%\VIAME-Base"

REM -------------------------------------------------------------------------------------------------------
REM Stage 2 - CUDA (adds CUDA/cuDNN support)
REM -------------------------------------------------------------------------------------------------------

XCOPY /E /I /Q "%VIAME_BUILD_DIR%\VIAME-Base" "%VIAME_INSTALL_DIR%"

CALL :BuildStage cuda "CUDA support"

CALL %~dp0build_common_functions.bat :CopyCuda12Dlls "%CUDA_ROOT%" "%VIAME_INSTALL_DIR%\bin"

CALL :DiffFiles files-base.txt files-cuda.txt diff-cuda.lst
"%ZIP_ROOT%\7z.exe" a -tzip "%VIAME_BUILD_DIR%\VIAME-CUDA.zip" @diff-cuda.lst

REM -------------------------------------------------------------------------------------------------------
REM Stage 3 - PyTorch (adds PyTorch + pytorch-libs)
REM -------------------------------------------------------------------------------------------------------

CALL :BuildStage pytorch "PyTorch support"

REM Remove duplicate CUDA libs from torch to save space
DEL "%VIAME_INSTALL_DIR%\%PYTHON_SUBDIR%\site-packages\torch\lib\cu*" 2>NUL

CALL :DiffFiles files-cuda.txt files-pytorch.txt diff-pytorch.lst
"%ZIP_ROOT%\7z.exe" a -tzip "%VIAME_BUILD_DIR%\VIAME-PyTorch.zip" @diff-pytorch.lst

REM -------------------------------------------------------------------------------------------------------
REM Stage 4 - VIVIA (adds VIVIA interface)
REM -------------------------------------------------------------------------------------------------------

CALL :BuildStage vivia "VIVIA interface"

CALL :DiffFiles files-pytorch.txt files-vivia.txt diff-vivia.lst
"%ZIP_ROOT%\7z.exe" a -tzip "%VIAME_BUILD_DIR%\VIAME-VIVIA.zip" @diff-vivia.lst

REM -------------------------------------------------------------------------------------------------------
REM Stage 5 - SEAL (adds SEAL toolkit)
REM -------------------------------------------------------------------------------------------------------

CALL :BuildStage seal "SEAL toolkit"

CALL :DiffFiles files-vivia.txt files-seal.txt diff-seal.lst
"%ZIP_ROOT%\7z.exe" a -tzip "%VIAME_BUILD_DIR%\VIAME-SEAL.zip" @diff-seal.lst

REM -------------------------------------------------------------------------------------------------------
REM Stage 6 - DIVE (adds DIVE interface)
REM -------------------------------------------------------------------------------------------------------

CALL :BuildStage dive "DIVE interface"

CALL :DiffFiles files-seal.txt files-dive.txt diff-dive.lst
"%ZIP_ROOT%\7z.exe" a -tzip "%VIAME_BUILD_DIR%\VIAME-DIVE.zip" @diff-dive.lst

MOVE "%VIAME_INSTALL_DIR%" "%VIAME_BUILD_DIR%\VIAME-Full"

REM -------------------------------------------------------------------------------------------------------
REM Build Complete
REM -------------------------------------------------------------------------------------------------------

ECHO.
ECHO ========================================
ECHO MSI Build Complete!
ECHO ========================================
ECHO.
ECHO Generated packages:
ECHO   - VIAME-Base.zip    (fletch + kwiver)
ECHO   - VIAME-CUDA.zip    (CUDA/cuDNN support)
ECHO   - VIAME-PyTorch.zip (PyTorch + libs)
ECHO   - VIAME-VIVIA.zip   (VIVIA interface)
ECHO   - VIAME-SEAL.zip    (SEAL toolkit)
ECHO   - VIAME-DIVE.zip    (DIVE interface)
ECHO.

GOTO :EOF

REM ==============================================================================
REM Local Subroutines
REM ==============================================================================

:BuildStage
REM Build a specific stage
REM %1 = stage name (base, cuda, pytorch, vivia, seal, dive)
REM %2 = display name for logging
ECHO.
ECHO ========================================
ECHO Building: %~2
ECHO ========================================
ECHO.

SET "VIAME_MSI_STAGE=%~1"

CALL %~dp0build_common_functions.bat :GenerateCTestDashboard build_server_windows_msi.cmake ctest_build_steps.cmake %VIAME_SOURCE_DIR%

"%CMAKE_ROOT%\bin\ctest.exe" -S %VIAME_SOURCE_DIR%\cmake\ctest_build_steps.cmake -VV
CALL %~dp0build_common_functions.bat :CheckCTestError "%~2"
GOTO :EOF

:SnapshotFiles
REM Create a file list snapshot, excluding include directory
REM %1 = output filename
powershell.exe "Get-ChildItem -Recurse '%VIAME_INSTALL_DIR%' | Resolve-Path -Relative" > tmp.txt
TYPE tmp.txt | findstr /v "install\include" > %~1
GOTO :EOF

:DiffFiles
REM Create a diff between two file lists
REM %1 = previous file list
REM %2 = current file list (will be created)
REM %3 = output diff list
CALL :SnapshotFiles %~2
FOR /f "delims=" %%A in (%~2) do @find "%%A" "%~1" >nul 2>nul || echo %%A>>%~3
GOTO :EOF
