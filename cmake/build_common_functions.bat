@ECHO OFF
REM ==============================================================================
REM VIAME Build Common Functions for Windows
REM Utility functions for all Windows build scripts
REM
REM Usage: CALL %~dp0build_common_functions.bat :FunctionName arg1 arg2 ...
REM ==============================================================================

REM Jump to the requested function if called with a label argument
IF NOT "%~1"=="" GOTO %~1
GOTO :EOF

REM ==============================================================================
REM CheckBuildDependencies
REM Verify that all required build tools are installed
REM Arguments:
REM   %2 = CMAKE_ROOT path
REM   %3 = GIT_ROOT path
REM   %4 = ZIP_ROOT path
REM   %5 = ZLIB_ROOT path
REM   %6 = CUDA_ROOT path (pass "SKIP" to skip CUDA check for CPU builds)
REM Returns: Sets ERRORLEVEL to 1 if any required dependency is missing
REM ==============================================================================
:CheckBuildDependencies
SETLOCAL EnableDelayedExpansion
SET "HAS_ERROR=0"

ECHO.
ECHO ========================================
ECHO Checking Build Dependencies
ECHO ========================================
ECHO.

REM Check CMake
IF NOT EXIST "%~2\bin\cmake.exe" (
    ECHO [MISSING] CMake not found at: %~2
    ECHO           Download from: https://cmake.org/download/
    SET "HAS_ERROR=1"
) ELSE (
    ECHO [OK] CMake: %~2
)

REM Check Git
IF NOT EXIST "%~3\cmd\git.exe" (
    ECHO [MISSING] Git not found at: %~3
    ECHO           Download from: https://git-scm.com/download/win
    SET "HAS_ERROR=1"
) ELSE (
    ECHO [OK] Git: %~3
)

REM Check 7-Zip
IF NOT EXIST "%~4\7z.exe" (
    ECHO [MISSING] 7-Zip not found at: %~4
    ECHO           Download from: https://www.7-zip.org/download.html
    SET "HAS_ERROR=1"
) ELSE (
    ECHO [OK] 7-Zip: %~4
)

REM Check ZLib
IF NOT EXIST "%~5\dll_x64\zlibwapi.dll" (
    ECHO [MISSING] ZLib not found at: %~5
    ECHO           Download from: https://www.zlib.net/
    SET "HAS_ERROR=1"
) ELSE (
    ECHO [OK] ZLib: %~5
)

REM Check CUDA (optional for CPU builds)
IF /I NOT "%~6"=="SKIP" (
    IF NOT EXIST "%~6\bin\nvcc.exe" (
        ECHO [MISSING] CUDA not found at: %~6
        ECHO           Download from: https://developer.nvidia.com/cuda-downloads
        SET "HAS_ERROR=1"
    ) ELSE (
        ECHO [OK] CUDA: %~6
        REM Check for cuDNN
        IF NOT EXIST "%~6\bin\cudnn64_9.dll" (
            IF NOT EXIST "%~6\bin\cudnn64_8.dll" (
                ECHO [WARNING] cuDNN not found in CUDA directory
                ECHO           Download from: https://developer.nvidia.com/cudnn
                ECHO           Extract to: %~6
            ) ELSE (
                ECHO [OK] cuDNN 8.x detected
            )
        ) ELSE (
            ECHO [OK] cuDNN 9.x detected
        )
    )
) ELSE (
    ECHO [SKIP] CUDA: Not required for CPU build
)

ECHO.

IF "!HAS_ERROR!"=="1" (
    ECHO ========================================
    ECHO ERROR: Missing required dependencies!
    ECHO ========================================
    ECHO.
    ECHO Please install the missing dependencies and try again.
    ECHO.
    ENDLOCAL
    EXIT /B 1
)

ECHO All dependencies found.
ECHO.
ENDLOCAL
EXIT /B 0

REM ==============================================================================
REM GenerateCTestDashboard
REM Generate CTest dashboard cmake file
REM Arguments:
REM   %2 = Platform cmake file to include (e.g., build_server_windows.cmake)
REM   %3 = Output file path (e.g., ctest_dashboard.cmake)
REM   %4 = Source directory containing cmake folder
REM ==============================================================================
:GenerateCTestDashboard
(
    ECHO # Auto-generated CTest dashboard file
    ECHO include^(${CMAKE_CURRENT_LIST_DIR}/%~2^)
    ECHO ctest_start^(${CTEST_BUILD_MODEL}^)
    ECHO ctest_configure^(BUILD ${CTEST_BINARY_DIRECTORY} SOURCE ${CTEST_SOURCE_DIRECTORY} OPTIONS "${OPTIONS}"^)
    ECHO ctest_build^(^)
    ECHO ctest_submit^(^)
) > %~4\cmake\%~3
GOTO :EOF

REM ==============================================================================
REM CopySystemDlls
REM Copy common system runtime DLLs to install directory
REM Arguments:
REM   %2 = Install bin directory
REM   %3 = VDIST_ROOT (Visual Studio OpenMP redistributable path)
REM   %4 = ZLIB_ROOT
REM   %5 = ZLIB_BUILD_DIR (optional - if provided, copies zlib1.dll from build)
REM ==============================================================================
:CopySystemDlls
COPY "%WINDIR%\System32\msvcr100.dll" "%~2" 2>NUL
COPY "%WINDIR%\System32\vcruntime140_1.dll" "%~2" 2>NUL
COPY "%~3\vcomp140.dll" "%~2" 2>NUL
COPY "%WINDIR%\SysWOW64\msvcr120.dll" "%~2" 2>NUL
COPY "%~4\dll_x64\zlibwapi.dll" "%~2" 2>NUL
IF NOT "%~5"=="" COPY "%~5\Release\zlib1.dll" "%~2" 2>NUL
GOTO :EOF

REM ==============================================================================
REM CopyCuda12Dlls
REM Copy CUDA 12.x and cuDNN 9.x DLLs to install directory
REM Arguments:
REM   %2 = CUDA root directory
REM   %3 = Install bin directory
REM ==============================================================================
:CopyCuda12Dlls
COPY "%~2\bin\cublas64_12.dll" "%~3" 2>NUL
COPY "%~2\bin\cublasLt64_12.dll" "%~3" 2>NUL
COPY "%~2\bin\cudart64_12.dll" "%~3" 2>NUL
COPY "%~2\bin\cudnn_adv64_9.dll" "%~3" 2>NUL
COPY "%~2\bin\cudnn_cnn64_9.dll" "%~3" 2>NUL
COPY "%~2\bin\cudnn_engines_precompiled64_9.dll" "%~3" 2>NUL
COPY "%~2\bin\cudnn_engines_runtime_compiled64_9.dll" "%~3" 2>NUL
COPY "%~2\bin\cudnn_graph64_9.dll" "%~3" 2>NUL
COPY "%~2\bin\cudnn_heuristic64_9.dll" "%~3" 2>NUL
COPY "%~2\bin\cudnn_ops64_9.dll" "%~3" 2>NUL
COPY "%~2\bin\cudnn64_9.dll" "%~3" 2>NUL
COPY "%~2\bin\cufft64_11.dll" "%~3" 2>NUL
COPY "%~2\bin\cufftw64_11.dll" "%~3" 2>NUL
COPY "%~2\bin\curand64_10.dll" "%~3" 2>NUL
COPY "%~2\bin\cusolver64_11.dll" "%~3" 2>NUL
COPY "%~2\bin\cusolverMg64_11.dll" "%~3" 2>NUL
COPY "%~2\bin\cusparse64_12.dll" "%~3" 2>NUL
GOTO :EOF

REM ==============================================================================
REM CreateZipPackage
REM Create zip package from install directory
REM Arguments:
REM   %2 = Install directory path
REM   %3 = Build directory path
REM   %4 = Output zip filename
REM   %5 = 7-Zip root directory
REM ==============================================================================
:CreateZipPackage
MOVE "%~2" "%~3\VIAME"
"%~5\7z.exe" a "%~3\%~4" "%~3\VIAME"
MOVE "%~3\VIAME" "%~2"
GOTO :EOF

REM ==============================================================================
REM CheckCTestError
REM Check if previous command failed and exit with error message
REM Arguments:
REM   %2 = Build stage name for error message
REM ==============================================================================
:CheckCTestError
IF %ERRORLEVEL% NEQ 0 (
    ECHO CTest build failed at %~2 with error code %ERRORLEVEL%
    EXIT /B %ERRORLEVEL%
)
GOTO :EOF
