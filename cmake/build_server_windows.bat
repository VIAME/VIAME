REM -------------------------------------------------------------------------------------------------------
REM CORE BUILD PIPELINE
REM -------------------------------------------------------------------------------------------------------

SET VIAME_SOURCE_DIR=C:\workspace\VIAME-Windows-GPU-Release
SET VIAME_BUILD_DIR=%VIAME_SOURCE_DIR%\build
SET VIAME_INSTALL_DIR=%VIAME_BUILD_DIR%\install

IF EXIST build rmdir /s /q build

IF NOT EXIST C:\tmp mkdir C:\tmp
IF EXIST C:\tmp\kv1 rmdir /s /q C:\tmp\kv1
IF EXIST C:\tmp\vm1 rmdir /s /q C:\tmp\vm1

SET "CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3"
SET "WIN32_ROOT=C:\Windows\System32"
Set "PYTHON_SUBDIR=lib\python3.6"
SET "PATH=%WIN32_ROOT%;C:\Windows;%WIN32_ROOT%\Wbem;%WIN32_ROOT%\WindowsPowerShell\v1.0;%WIN32_ROOT%\OpenSSH"
SET "PATH=%CUDA_ROOT%\bin;%CUDA_ROOT%\libnvvp;C:\Program Files (x86)\NVIDIA Corporation\PhysX\Common;C:\Program Files\NVIDIA Corporation\NVIDIA NvDLISR;%PATH%"
SET "PATH=C:\Program Files\Git\cmd;C:\Program Files\CMake\bin;%PATH%"
SET "PYTHONPATH=%VIAME_INSTALL_DIR%\%PYTHON_SUBDIR%;%VIAME_INSTALL_DIR%\%PYTHON_SUBDIR%\site-packages"

git submodule update --init --recursive

"C:\Program Files\CMake\bin\ctest.exe" -S jenkins_dashboard.cmake -VV

REM -------------------------------------------------------------------------------------------------------
REM HACKS UNTIL THESE THINGS ARE BETTER HANDLED IN CODE
REM -------------------------------------------------------------------------------------------------------

SET MISSING_SVM_DLL=%VIAME_SOURCE_DIR%\packages\smqtk\TPL\libsvm-3.1-custom\libsvm.dll
SET MISSING_DNET_EXE=%VIAME_BUILD_DIR%\build\src\darknet-build\Release\darknet.exe

MOVE %MISSING_SVM_DLL% %VIAME_INSTALL_DIR%\bin
MOVE %MISSING_DNET_EXE% %VIAME_INSTALL_DIR%\bin

COPY %WIN32_ROOT%\msvcr100.dll %VIAME_INSTALL_DIR%\bin
COPY %WIN32_ROOT%\vcruntime140_1.dll %VIAME_INSTALL_DIR%\bin
COPY C:\Windows\SysWOW64\msvcr120.dll %VIAME_INSTALL_DIR%\bin
COPY %VIAME_SOURCE_DIR%\packages\darknet\3rdparty\pthreads\bin\pthreadVC2.dll %VIAME_INSTALL_DIR%\bin
COPY "C:\Program Files\ZLib\dll_x64\zlibwapi.dll" %VIAME_INSTALL_DIR%\bin

DEL "%VIAME_INSTALL_DIR%\%PYTHON_SUBDIR%\site-packages\torch\lib\cu*"

COPY "%CUDA_ROOT%\bin\cublas64_11.dll" %VIAME_INSTALL_DIR%\bin
COPY "%CUDA_ROOT%\bin\cublasLt64_11.dll" %VIAME_INSTALL_DIR%\bin
COPY "%CUDA_ROOT%\bin\cudart64_110.dll" %VIAME_INSTALL_DIR%\bin
COPY "%CUDA_ROOT%\bin\cudnn_adv_infer64_8.dll" %VIAME_INSTALL_DIR%\bin
COPY "%CUDA_ROOT%\bin\cudnn_adv_train64_8.dll" %VIAME_INSTALL_DIR%\bin
COPY "%CUDA_ROOT%\bin\cudnn_cnn_infer64_8.dll" %VIAME_INSTALL_DIR%\bin
COPY "%CUDA_ROOT%\bin\cudnn_cnn_train64_8.dll" %VIAME_INSTALL_DIR%\bin
COPY "%CUDA_ROOT%\bin\cudnn_ops_infer64_8.dll" %VIAME_INSTALL_DIR%\bin
COPY "%CUDA_ROOT%\bin\cudnn_ops_train64_8.dll" %VIAME_INSTALL_DIR%\bin
COPY "%CUDA_ROOT%\bin\cudnn64_8.dll" %VIAME_INSTALL_DIR%\bin
COPY "%CUDA_ROOT%\bin\cufft64_10.dll" %VIAME_INSTALL_DIR%\bin
COPY "%CUDA_ROOT%\bin\cufftw64_10.dll" %VIAME_INSTALL_DIR%\bin
COPY "%CUDA_ROOT%\bin\curand64_10.dll" %VIAME_INSTALL_DIR%\bin
COPY "%CUDA_ROOT%\bin\cusolver64_11.dll" %VIAME_INSTALL_DIR%\bin
COPY "%CUDA_ROOT%\bin\cusolverMg64_11.dll" %VIAME_INSTALL_DIR%\bin
COPY "%CUDA_ROOT%\bin\cusparse64_11.dll" %VIAME_INSTALL_DIR%\bin

REM -------------------------------------------------------------------------------------------------------
REM COMPRESS FINAL PACKAGE
REM -------------------------------------------------------------------------------------------------------

MOVE "%VIAME_INSTALL_DIR%" "%VIAME_BUILD_DIR%\VIAME"

"C:\Program Files\7-Zip\7z.exe" a "%VIAME_BUILD_DIR%/VIAME-v1.0.0-Windows-64Bit.zip" "%VIAME_BUILD_DIR%/VIAME

MOVE "%VIAME_BUILD_DIR%\VIAME" "%VIAME_INSTALL_DIR%"
