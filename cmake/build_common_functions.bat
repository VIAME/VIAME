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
