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
REM   %1 = Platform cmake file to include (e.g., build_server_windows.cmake)
REM   %2 = Output file path (e.g., ctest_dashboard.cmake)
REM   %3 = Source directory containing cmake folder
REM ==============================================================================
:GenerateCTestDashboard
(
    ECHO # Auto-generated CTest dashboard file
    ECHO include^(${CMAKE_CURRENT_LIST_DIR}/%~1^)
    ECHO ctest_start^(${CTEST_BUILD_MODEL}^)
    ECHO ctest_configure^(BUILD ${CTEST_BINARY_DIRECTORY} SOURCE ${CTEST_SOURCE_DIRECTORY} OPTIONS "${OPTIONS}"^)
    ECHO ctest_build^(^)
    ECHO ctest_submit^(^)
) > %~3\cmake\%~2
GOTO :EOF

REM ==============================================================================
REM CopySystemDlls
REM Copy common system runtime DLLs to install directory
REM Arguments:
REM   %1 = Install bin directory
REM   %2 = VDIST_ROOT (Visual Studio OpenMP redistributable path)
REM   %3 = ZLIB_ROOT
REM   %4 = ZLIB_BUILD_DIR (optional - if provided, copies zlib1.dll from build)
REM ==============================================================================
:CopySystemDlls
COPY "%WINDIR%\System32\msvcr100.dll" "%~1" 2>NUL
COPY "%WINDIR%\System32\vcruntime140_1.dll" "%~1" 2>NUL
COPY "%~2\vcomp140.dll" "%~1" 2>NUL
COPY "%WINDIR%\SysWOW64\msvcr120.dll" "%~1" 2>NUL
COPY "%~3\dll_x64\zlibwapi.dll" "%~1" 2>NUL
IF NOT "%~4"=="" COPY "%~4\Release\zlib1.dll" "%~1" 2>NUL
GOTO :EOF

REM ==============================================================================
REM CopyCuda12Dlls
REM Copy CUDA 12.x and cuDNN 9.x DLLs to install directory
REM Arguments:
REM   %1 = CUDA root directory
REM   %2 = Install bin directory
REM ==============================================================================
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

REM ==============================================================================
REM CreateZipPackage
REM Create zip package from install directory
REM Arguments:
REM   %1 = Install directory path
REM   %2 = Build directory path
REM   %3 = Output zip filename
REM   %4 = 7-Zip root directory
REM ==============================================================================
:CreateZipPackage
MOVE "%~1" "%~2\VIAME"
"%~4\7z.exe" a "%~2\%~3" "%~2\VIAME"
MOVE "%~2\VIAME" "%~1"
GOTO :EOF

REM ==============================================================================
REM CheckCTestError
REM Check if previous command failed and exit with error message
REM Arguments:
REM   %1 = Build stage name for error message
REM ==============================================================================
:CheckCTestError
IF %ERRORLEVEL% NEQ 0 (
    ECHO CTest build failed at %~1 with error code %ERRORLEVEL%
    EXIT /B %ERRORLEVEL%
)
GOTO :EOF
