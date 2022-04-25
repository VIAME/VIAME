REM ------------------------------------------------------------------------------------------------------------------------------------------------------------
REM CORE BUILD PIPELINE
REM ------------------------------------------------------------------------------------------------------------------------------------------------------------

SET VIAME_SOURCE_DIR=C:\workspace\VIAME-Windows-CPU-Release
SET VIAME_BUILD_DIR=%VIAME_SOURCE_DIR%\build

IF EXIST build rmdir /s /q build

IF NOT EXIST C:\tmp mkdir C:\tmp
IF EXIST C:\tmp\kv2 rmdir /s /q C:\tmp\kv2
IF EXIST C:\tmp\vm2 rmdir /s /q C:\tmp\vm2

SET "WIN32_ROOT=C:\Windows\System32"
SET "PATH=%WIN32_ROOT%;C:\Windows;%WIN32_ROOT%\Wbem;%WIN32_ROOT%\WindowsPowerShell\v1.0;%WIN32_ROOT%\OpenSSH"
SET "PATH=C:\Program Files\Git\cmd;C:\Program Files\CMake\bin;%PATH%"

SET "PYTHONPATH=%VIAME_BUILD_DIR%\install\lib\python3.8;%VIAME_BUILD_DIR%\install\lib\python3.8\site-packages"

git submodule update --init --recursive

"C:\Program Files\CMake\bin\ctest.exe" -S jenkins_dashboard.cmake -VV

REM ------------------------------------------------------------------------------------------------------------------------------------------------------------
REM HACKS UNTIL THESE THINGS ARE BETTER HANDLED IN CODE
REM ------------------------------------------------------------------------------------------------------------------------------------------------------------

MOVE "%VIAME_BUILD_DIR%\install" "%VIAME_BUILD_DIR%\VIAME"

SET MISSING_SVM_DLL=%VIAME_SOURCE_DIR%\packages\smqtk\TPL\libsvm-3.1-custom\libsvm.dll
SET MISSING_DNET_EXE=%VIAME_BUILD_DIR%\build\src\darknet-build\Release\darknet.exe

MOVE %MISSING_SVM_DLL% %VIAME_BUILD_DIR%\VIAME\bin
MOVE %MISSING_DNET_EXE% %VIAME_BUILD_DIR%\VIAME\bin

COPY /y "%VIAME_SOURCE_DIR%\cmake\setup_viame.bat.install" %VIAME_BUILD_DIR%\VIAME\setup_viame.bat

XCOPY /E /I "%VIAME_SOURCE_DIR%\packages\patches\pycocotools" %VIAME_BUILD_DIR%\VIAME\lib\python3.8\site-packages

COPY %WIN32_ROOT%\msvcr100.dll %VIAME_BUILD_DIR%\VIAME\bin
COPY %WIN32_ROOT%\vcruntime140_1.dll %VIAME_BUILD_DIR%\VIAME\bin
COPY C:\Windows\SysWOW64\msvcr120.dll %VIAME_BUILD_DIR%\VIAME\bin
COPY %VIAME_SOURCE_DIR%\packages\darknet\3rdparty\pthreads\pthreadVC2.dll %VIAME_BUILD_DIR%\VIAME\bin

REM ------------------------------------------------------------------------------------------------------------------------------------------------------------
REM COMPRESS FINAL PACKAGE
REM ------------------------------------------------------------------------------------------------------------------------------------------------------------

"C:\Program Files\7-Zip\7z.exe" a "%VIAME_BUILD_DIR%/VIAME-v1.0.0-Windows-64Bit-CPU.zip" "%VIAME_BUILD_DIR%/VIAME
