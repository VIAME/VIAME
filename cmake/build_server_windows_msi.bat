REM -------------------------------------------------------------------------------------------------------
REM VIAME Windows MSI Build Script
REM
REM Builds VIAME in stages to create separate installable packages:
REM   1. Core        - fletch + kwiver + vxl + opencv + python (CPU only)
REM   2. CUDA        - adds CUDA/cuDNN support + DLLs
REM   3. PyTorch     - adds PyTorch + all pytorch-libs
REM   4. Extra-CPP   - adds Darknet, SVM, PostgreSQL
REM   5. DIVE        - adds DIVE GUI
REM   6. VIVIA       - adds VIVIA interface (Qt, VTK, GDAL)
REM   7. SEAL        - adds SEAL toolkit
REM   8. Models      - adds model downloads
REM   9. Dev-Headers - include and share folders (development headers)
REM  10. WiX Build   - builds the network installer bundle (requires WiX v5)
REM
REM Note: Stages 1-8 exclude include/ and share/ folders; Stage 9 contains only those
REM       Stage 10 builds the WiX installer (runs automatically, requires WiX v5 CLI)
REM
REM Usage: build_server_windows_msi.bat [start_stage]
REM   start_stage - Optional stage number (1-9) to resume from. Default is 1.
REM   Example: build_server_windows_msi.bat 3  (resumes from PyTorch stage)
REM
REM Uses VIAME_MSI_STAGE environment variable to control cmake configuration
REM -------------------------------------------------------------------------------------------------------

@ECHO OFF
SETLOCAL EnableDelayedExpansion

REM Parse command line argument for resume capability
SET "START_STAGE=1"
IF NOT "%~1"=="" SET "START_STAGE=%~1"

REM Validate start stage
IF %START_STAGE% LSS 1 (
    ECHO ERROR: Invalid start stage %START_STAGE%. Must be between 1 and 9.
    EXIT /B 1
)
IF %START_STAGE% GTR 9 (
    ECHO ERROR: Invalid start stage %START_STAGE%. Must be between 1 and 9.
    EXIT /B 1
)

ECHO.
ECHO ========================================
ECHO VIAME MSI Build - Starting from Stage %START_STAGE%
ECHO ========================================
ECHO.

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
REM Check Build Dependencies
REM -------------------------------------------------------------------------------------------------------

CALL %~dp0build_common_functions.bat :CheckBuildDependencies "%CMAKE_ROOT%" "%GIT_ROOT%" "%ZIP_ROOT%" "%ZLIB_ROOT%" "%CUDA_ROOT%"
IF ERRORLEVEL 1 EXIT /B 1

REM -------------------------------------------------------------------------------------------------------
REM Initialize Build Environment
REM -------------------------------------------------------------------------------------------------------

IF %START_STAGE% EQU 1 (
    ECHO Initializing fresh build environment...
    IF EXIST build rmdir /s /q build

    IF NOT EXIST C:\tmp mkdir C:\tmp
    IF EXIST C:\tmp\fl5 rmdir /s /q C:\tmp\fl5
    IF EXIST C:\tmp\kv5 rmdir /s /q C:\tmp\kv5
    IF EXIST C:\tmp\vm5 rmdir /s /q C:\tmp\vm5

    git config --system core.longpaths true
    git submodule update --init --recursive
) ELSE (
    ECHO Resuming from stage %START_STAGE%, skipping initialization...
    REM Restore install directory from previous stage if resuming
    IF %START_STAGE% EQU 2 IF EXIST "%VIAME_BUILD_DIR%\VIAME-Core" XCOPY /E /I /Q "%VIAME_BUILD_DIR%\VIAME-Core" "%VIAME_INSTALL_DIR%"
    IF %START_STAGE% GTR 2 IF EXIST "%VIAME_BUILD_DIR%\VIAME-Full" XCOPY /E /I /Q "%VIAME_BUILD_DIR%\VIAME-Full" "%VIAME_INSTALL_DIR%"
)

REM -------------------------------------------------------------------------------------------------------
REM Stage 1 - Core (fletch + kwiver + vxl + opencv + python, CPU only)
REM -------------------------------------------------------------------------------------------------------

IF %START_STAGE% LEQ 1 (
    CALL :BuildStage core "Core (fletch + kwiver, CPU only)"
    IF ERRORLEVEL 1 GOTO :BuildFailed

    CALL %~dp0build_common_functions.bat :CopySystemDlls "%VIAME_INSTALL_DIR%\bin" "%WIN64_ROOT%" "%ZLIB_ROOT%"

    CALL :SnapshotFiles files-core.txt
    "%ZIP_ROOT%\7z.exe" a -tzip "%VIAME_BUILD_DIR%\VIAME-Core.zip" @files-core.txt
    IF ERRORLEVEL 1 GOTO :ZipFailed
    CALL :ReportPackageSize "VIAME-Core.zip"

    MOVE "%VIAME_INSTALL_DIR%" "%VIAME_BUILD_DIR%\VIAME-Core"
)

REM -------------------------------------------------------------------------------------------------------
REM Stage 2 - CUDA (adds CUDA/cuDNN support + DLLs)
REM -------------------------------------------------------------------------------------------------------

IF %START_STAGE% LEQ 2 (
    IF %START_STAGE% EQU 2 XCOPY /E /I /Q "%VIAME_BUILD_DIR%\VIAME-Core" "%VIAME_INSTALL_DIR%"

    CALL :BuildStage cuda "CUDA/cuDNN support"
    IF ERRORLEVEL 1 GOTO :BuildFailed

    CALL %~dp0build_common_functions.bat :CopyCuda12Dlls "%CUDA_ROOT%" "%VIAME_INSTALL_DIR%\bin"

    CALL :DiffFiles files-core.txt files-cuda.txt diff-cuda.lst
    "%ZIP_ROOT%\7z.exe" a -tzip "%VIAME_BUILD_DIR%\VIAME-CUDA.zip" @diff-cuda.lst
    IF ERRORLEVEL 1 GOTO :ZipFailed
    CALL :ReportPackageSize "VIAME-CUDA.zip"
)

REM -------------------------------------------------------------------------------------------------------
REM Stage 3 - PyTorch (adds PyTorch + all pytorch-libs)
REM -------------------------------------------------------------------------------------------------------

IF %START_STAGE% LEQ 3 (
    CALL :BuildStage pytorch "PyTorch + pytorch-libs"
    IF ERRORLEVEL 1 GOTO :BuildFailed

    REM Remove duplicate CUDA libs from torch to save space
    DEL "%VIAME_INSTALL_DIR%\%PYTHON_SUBDIR%\site-packages\torch\lib\cu*" 2>NUL

    CALL :DiffFiles files-cuda.txt files-pytorch.txt diff-pytorch.lst
    "%ZIP_ROOT%\7z.exe" a -tzip "%VIAME_BUILD_DIR%\VIAME-PyTorch.zip" @diff-pytorch.lst
    IF ERRORLEVEL 1 GOTO :ZipFailed
    CALL :ReportPackageSize "VIAME-PyTorch.zip"
)

REM -------------------------------------------------------------------------------------------------------
REM Stage 4 - Extra-CPP (adds Darknet, SVM, PostgreSQL)
REM -------------------------------------------------------------------------------------------------------

IF %START_STAGE% LEQ 4 (
    CALL :BuildStage extra-cpp "Extra CPP (Darknet, SVM, PostgreSQL)"
    IF ERRORLEVEL 1 GOTO :BuildFailed

    CALL :DiffFiles files-pytorch.txt files-extra-cpp.txt diff-extra-cpp.lst
    "%ZIP_ROOT%\7z.exe" a -tzip "%VIAME_BUILD_DIR%\VIAME-Extra-CPP.zip" @diff-extra-cpp.lst
    IF ERRORLEVEL 1 GOTO :ZipFailed
    CALL :ReportPackageSize "VIAME-Extra-CPP.zip"
)

REM -------------------------------------------------------------------------------------------------------
REM Stage 5 - DIVE (adds DIVE GUI)
REM -------------------------------------------------------------------------------------------------------

IF %START_STAGE% LEQ 5 (
    CALL :BuildStage dive "DIVE GUI"
    IF ERRORLEVEL 1 GOTO :BuildFailed

    CALL :DiffFiles files-extra-cpp.txt files-dive.txt diff-dive.lst
    "%ZIP_ROOT%\7z.exe" a -tzip "%VIAME_BUILD_DIR%\VIAME-DIVE.zip" @diff-dive.lst
    IF ERRORLEVEL 1 GOTO :ZipFailed
    CALL :ReportPackageSize "VIAME-DIVE.zip"
)

REM -------------------------------------------------------------------------------------------------------
REM Stage 6 - VIVIA (adds VIVIA interface with Qt, VTK, GDAL)
REM -------------------------------------------------------------------------------------------------------

IF %START_STAGE% LEQ 6 (
    CALL :BuildStage vivia "VIVIA interface"
    IF ERRORLEVEL 1 GOTO :BuildFailed

    CALL :DiffFiles files-dive.txt files-vivia.txt diff-vivia.lst
    "%ZIP_ROOT%\7z.exe" a -tzip "%VIAME_BUILD_DIR%\VIAME-VIVIA.zip" @diff-vivia.lst
    IF ERRORLEVEL 1 GOTO :ZipFailed
    CALL :ReportPackageSize "VIAME-VIVIA.zip"
)

REM -------------------------------------------------------------------------------------------------------
REM Stage 7 - SEAL (adds SEAL toolkit)
REM -------------------------------------------------------------------------------------------------------

IF %START_STAGE% LEQ 7 (
    CALL :BuildStage seal "SEAL toolkit"
    IF ERRORLEVEL 1 GOTO :BuildFailed

    CALL :DiffFiles files-vivia.txt files-seal.txt diff-seal.lst
    "%ZIP_ROOT%\7z.exe" a -tzip "%VIAME_BUILD_DIR%\VIAME-SEAL.zip" @diff-seal.lst
    IF ERRORLEVEL 1 GOTO :ZipFailed
    CALL :ReportPackageSize "VIAME-SEAL.zip"
)

REM -------------------------------------------------------------------------------------------------------
REM Stage 8 - Models (adds model downloads)
REM -------------------------------------------------------------------------------------------------------

IF %START_STAGE% LEQ 8 (
    CALL :BuildStage models "Model downloads"
    IF ERRORLEVEL 1 GOTO :BuildFailed

    CALL :DiffFiles files-seal.txt files-models.txt diff-models.lst
    "%ZIP_ROOT%\7z.exe" a -tzip "%VIAME_BUILD_DIR%\VIAME-Models.zip" @diff-models.lst
    IF ERRORLEVEL 1 GOTO :ZipFailed
    CALL :ReportPackageSize "VIAME-Models.zip"
)

REM -------------------------------------------------------------------------------------------------------
REM Stage 9 - Dev-Headers (include and share folders for development)
REM -------------------------------------------------------------------------------------------------------

IF %START_STAGE% LEQ 9 (
    CALL :SnapshotDevHeaders files-dev-headers.txt
    "%ZIP_ROOT%\7z.exe" a -tzip "%VIAME_BUILD_DIR%\VIAME-Dev-Headers.zip" @files-dev-headers.txt
    IF ERRORLEVEL 1 GOTO :ZipFailed
    CALL :ReportPackageSize "VIAME-Dev-Headers.zip"

    MOVE "%VIAME_INSTALL_DIR%" "%VIAME_BUILD_DIR%\VIAME-Full"
)

REM -------------------------------------------------------------------------------------------------------
REM Stage 10 - Build WiX Network Installer
REM -------------------------------------------------------------------------------------------------------

ECHO.
ECHO ========================================
ECHO Building WiX Network Installer
ECHO ========================================
ECHO.

REM Generate UI theme from CSV (model checkboxes)
ECHO Generating installer UI from model addons CSV...
python "%~dp0msi_generate_installer.py" --ui-only
IF ERRORLEVEL 1 (
    ECHO WARNING: Failed to generate UI theme. Continuing with existing theme...
)

REM Check if wix CLI is available
WHERE wix >NUL 2>&1
IF ERRORLEVEL 1 (
    ECHO.
    ECHO WARNING: WiX CLI tool not found. Install with:
    ECHO   dotnet tool install --global wix
    ECHO   wix extension add WixToolset.Bal.wixext
    ECHO.
    ECHO Skipping network installer build...
    GOTO :SkipWixBuild
)

REM Build the network installer bundle
ECHO Building network installer with WiX v5...
pushd "%~dp0"
wix build -ext WixToolset.Bal.wixext msi_viame_network.wxs -o "%VIAME_BUILD_DIR%\viame-installer.exe"
IF ERRORLEVEL 1 (
    ECHO.
    ECHO WARNING: WiX build failed. Check WiX installation and extension.
    popd
    GOTO :SkipWixBuild
)
popd

ECHO Network installer created: %VIAME_BUILD_DIR%\viame-installer.exe
FOR %%F IN ("%VIAME_BUILD_DIR%\viame-installer.exe") DO (
    CALL :FormatSize "%%~zF" "viame-installer.exe"
)

:SkipWixBuild

REM -------------------------------------------------------------------------------------------------------
REM Build Complete
REM -------------------------------------------------------------------------------------------------------

ECHO.
ECHO ========================================
ECHO MSI Build Complete!
ECHO ========================================
ECHO.
ECHO Generated packages:
ECHO   - VIAME-Core.zip        (fletch + kwiver + vxl + opencv + python, CPU only)
ECHO   - VIAME-CUDA.zip        (CUDA/cuDNN support + DLLs)
ECHO   - VIAME-PyTorch.zip     (PyTorch + all pytorch-libs)
ECHO   - VIAME-Extra-CPP.zip   (Darknet, SVM, PostgreSQL)
ECHO   - VIAME-DIVE.zip        (DIVE GUI)
ECHO   - VIAME-VIVIA.zip       (VIVIA interface with Qt, VTK, GDAL)
ECHO   - VIAME-SEAL.zip        (SEAL toolkit)
ECHO   - VIAME-Models.zip      (Model downloads)
ECHO   - VIAME-Dev-Headers.zip (include + share folders for development)
ECHO   - viame-installer.exe   (WiX network installer bundle)
ECHO.
ECHO Package sizes:
FOR %%F IN ("%VIAME_BUILD_DIR%\VIAME-*.zip") DO (
    CALL :FormatSize "%%~zF" "%%~nxF"
)
IF EXIST "%VIAME_BUILD_DIR%\viame-installer.exe" (
    FOR %%F IN ("%VIAME_BUILD_DIR%\viame-installer.exe") DO (
        CALL :FormatSize "%%~zF" "%%~nxF"
    )
)
ECHO.

GOTO :EOF

REM -------------------------------------------------------------------------------------------------------
REM Error Handlers
REM -------------------------------------------------------------------------------------------------------

:BuildFailed
ECHO.
ECHO ========================================
ECHO BUILD FAILED!
ECHO ========================================
ECHO.
ECHO The build failed at stage: %VIAME_MSI_STAGE%
ECHO To resume, run: %~nx0 [stage_number]
ECHO.
EXIT /B 1

:ZipFailed
ECHO.
ECHO ========================================
ECHO ZIP PACKAGING FAILED!
ECHO ========================================
ECHO.
ECHO Failed to create zip package.
ECHO.
EXIT /B 1

REM ==============================================================================
REM Local Subroutines
REM ==============================================================================

:BuildStage
REM Build a specific stage
REM %1 = stage name (core, cuda, pytorch, extra-cpp, dive, vivia, seal, models)
REM %2 = display name for logging
ECHO.
ECHO ========================================
ECHO Building: %~2
ECHO ========================================
ECHO.

SET "VIAME_MSI_STAGE=%~1"

CALL %~dp0build_common_functions.bat :GenerateCTestDashboard build_server_windows_msi.cmake ctest_build_steps.cmake %VIAME_SOURCE_DIR%

"%CMAKE_ROOT%\bin\ctest.exe" -S %VIAME_SOURCE_DIR%\cmake\ctest_build_steps.cmake -VV
IF ERRORLEVEL 1 EXIT /B 1
GOTO :EOF

:SnapshotFiles
REM Create a file list snapshot, excluding include and share directories
REM %1 = output filename
powershell.exe -NoProfile -Command "Get-ChildItem -Recurse '%VIAME_INSTALL_DIR%' -File | Resolve-Path -Relative | Where-Object { $_ -notmatch 'install\\include' -and $_ -notmatch 'install\\share' }" > %~1
GOTO :EOF

:SnapshotDevHeaders
REM Create a file list snapshot of ONLY include and share directories
REM %1 = output filename
powershell.exe -NoProfile -Command "Get-ChildItem -Recurse '%VIAME_INSTALL_DIR%' -File | Resolve-Path -Relative | Where-Object { $_ -match 'install\\include' -or $_ -match 'install\\share' }" > %~1
GOTO :EOF

:DiffFiles
REM Create a diff between two file lists (optimized PowerShell version)
REM %1 = previous file list
REM %2 = current file list (will be created)
REM %3 = output diff list
CALL :SnapshotFiles %~2
powershell.exe -NoProfile -Command "$prev = Get-Content '%~1'; $curr = Get-Content '%~2'; $curr | Where-Object { $prev -notcontains $_ }" > %~3
GOTO :EOF

:ReportPackageSize
REM Report the size of a package immediately after creation
REM %1 = package filename (just the name, not full path)
FOR %%F IN ("%VIAME_BUILD_DIR%\%~1") DO (
    CALL :FormatSize "%%~zF" "%~1"
)
GOTO :EOF

:FormatSize
REM Format and display file size in human-readable format
REM %1 = size in bytes
REM %2 = filename
SET "SIZE=%~1"
SET "NAME=%~2"
IF "%SIZE%"=="" (
    ECHO   %NAME%: [not found]
    GOTO :EOF
)
SET /A "SIZE_MB=%SIZE% / 1048576"
SET /A "SIZE_GB=%SIZE% / 1073741824"
IF %SIZE_GB% GEQ 1 (
    SET /A "SIZE_GB_DEC=(%SIZE% %% 1073741824) * 10 / 1073741824"
    ECHO   %NAME%: %SIZE_GB%.!SIZE_GB_DEC! GB
) ELSE (
    ECHO   %NAME%: %SIZE_MB% MB
)
IF %SIZE_MB% GEQ 2048 (
    ECHO   WARNING: %NAME% exceeds 2 GB MSI limit!
)
GOTO :EOF
