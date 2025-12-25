@ECHO OFF
REM ==============================================================================
REM VIAME Build Common Functions for Windows
REM Utility functions for all Windows build scripts
REM
REM Usage: CALL cmake\build_common_functions.bat
REM ==============================================================================

GOTO :EOF

REM ==============================================================================
REM Generate CTest dashboard cmake file
REM Arguments:
REM   %1 = Platform cmake file to include (e.g., build_server_windows.cmake)
REM   %2 = Output file path (e.g., ctest_dashboard.cmake)
REM ==============================================================================
:GenerateCTestDashboard
SETLOCAL
SET "PLATFORM_CMAKE=%~1"
SET "OUTPUT_FILE=%~2"

(
    ECHO # Auto-generated CTest dashboard file
    ECHO # Platform: %PLATFORM_CMAKE%
    ECHO.
    ECHO # Get platform specific build info
    ECHO include^(%PLATFORM_CMAKE%^)
    ECHO.
    ECHO # Run CTest
    ECHO ctest_start^(${CTEST_BUILD_MODEL}^)
    ECHO ctest_configure^(BUILD ${CTEST_BINARY_DIRECTORY} SOURCE ${CTEST_SOURCE_DIRECTORY}
    ECHO                 OPTIONS "${OPTIONS}"^)
    ECHO ctest_build^(^)
    ECHO ctest_submit^(^)
) > "%OUTPUT_FILE%"

ECHO Generated CTest dashboard: %OUTPUT_FILE%
ENDLOCAL
GOTO :EOF

REM ==============================================================================
REM Extract VIAME version from RELEASE_NOTES.md
REM Arguments:
REM   %1 = Source directory containing RELEASE_NOTES.md
REM Sets: VIAME_VERSION environment variable
REM ==============================================================================
:ExtractVersion
SETLOCAL ENABLEDELAYEDEXPANSION
SET "SOURCE_DIR=%~1"
FOR /f "tokens=1 delims= " %%a IN (%SOURCE_DIR%\RELEASE_NOTES.md) DO (
    ENDLOCAL
    SET "VIAME_VERSION=%%a"
    GOTO :EOF
)
GOTO :EOF

REM ==============================================================================
REM Copy CUDA DLLs to install directory
REM Arguments:
REM   %1 = CUDA root directory
REM   %2 = Install bin directory
REM ==============================================================================
:CopyCudaDlls
SETLOCAL
SET "CUDA_ROOT=%~1"
SET "INSTALL_BIN=%~2"

COPY "%CUDA_ROOT%\bin\cublas64_12.dll" "%INSTALL_BIN%" 2>NUL
COPY "%CUDA_ROOT%\bin\cublasLt64_12.dll" "%INSTALL_BIN%" 2>NUL
COPY "%CUDA_ROOT%\bin\cudart64_12.dll" "%INSTALL_BIN%" 2>NUL
COPY "%CUDA_ROOT%\bin\cudnn_adv64_9.dll" "%INSTALL_BIN%" 2>NUL
COPY "%CUDA_ROOT%\bin\cudnn_cnn64_9.dll" "%INSTALL_BIN%" 2>NUL
COPY "%CUDA_ROOT%\bin\cudnn_engines_precompiled64_9.dll" "%INSTALL_BIN%" 2>NUL
COPY "%CUDA_ROOT%\bin\cudnn_engines_runtime_compiled64_9.dll" "%INSTALL_BIN%" 2>NUL
COPY "%CUDA_ROOT%\bin\cudnn_graph64_9.dll" "%INSTALL_BIN%" 2>NUL
COPY "%CUDA_ROOT%\bin\cudnn_heuristic64_9.dll" "%INSTALL_BIN%" 2>NUL
COPY "%CUDA_ROOT%\bin\cudnn_ops64_9.dll" "%INSTALL_BIN%" 2>NUL
COPY "%CUDA_ROOT%\bin\cudnn64_9.dll" "%INSTALL_BIN%" 2>NUL
COPY "%CUDA_ROOT%\bin\cufft64_11.dll" "%INSTALL_BIN%" 2>NUL
COPY "%CUDA_ROOT%\bin\cufftw64_11.dll" "%INSTALL_BIN%" 2>NUL
COPY "%CUDA_ROOT%\bin\curand64_10.dll" "%INSTALL_BIN%" 2>NUL
COPY "%CUDA_ROOT%\bin\cusolver64_11.dll" "%INSTALL_BIN%" 2>NUL
COPY "%CUDA_ROOT%\bin\cusolverMg64_11.dll" "%INSTALL_BIN%" 2>NUL
COPY "%CUDA_ROOT%\bin\cusparse64_12.dll" "%INSTALL_BIN%" 2>NUL

ENDLOCAL
GOTO :EOF

REM ==============================================================================
REM Copy system runtime DLLs to install directory
REM Arguments:
REM   %1 = Install bin directory
REM   %2 = VDIST_ROOT (Visual Studio OpenMP redistributable path)
REM   %3 = ZLIB_ROOT
REM   %4 = ZLIB_BUILD_DIR
REM ==============================================================================
:CopySystemDlls
SETLOCAL
SET "INSTALL_BIN=%~1"
SET "VDIST_ROOT=%~2"
SET "ZLIB_ROOT=%~3"
SET "ZLIB_BUILD_DIR=%~4"

COPY "%WINDIR%\System32\msvcr100.dll" "%INSTALL_BIN%" 2>NUL
COPY "%WINDIR%\System32\vcruntime140_1.dll" "%INSTALL_BIN%" 2>NUL
COPY "%VDIST_ROOT%\vcomp140.dll" "%INSTALL_BIN%" 2>NUL
COPY "%WINDIR%\SysWOW64\msvcr120.dll" "%INSTALL_BIN%" 2>NUL
COPY "%ZLIB_ROOT%\dll_x64\zlibwapi.dll" "%INSTALL_BIN%" 2>NUL
COPY "%ZLIB_BUILD_DIR%\Release\zlib1.dll" "%INSTALL_BIN%" 2>NUL

ENDLOCAL
GOTO :EOF
