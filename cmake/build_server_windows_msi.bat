REM ---------------------------------------------------
REM Round1 - VIAME Core
REM ---------------------------------------------------

SET VIAME_SOURCE_DIR=C:\workspace\VIAME-Windows-GPU-MSI
SET VIAME_BUILD_DIR=%VIAME_SOURCE_DIR%\build

IF EXIST build rmdir /s /q build

IF NOT EXIST C:\tmp mkdir C:\tmp
IF EXIST C:\tmp\kv5 rmdir /s /q C:\tmp\kv5
IF EXIST C:\tmp\vm6 rmdir /s /q C:\tmp\vm5

SET "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\libnvvp;C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0\;C:\Program Files (x86)\NVIDIA Corporation\PhysX\Common;C:\Program Files\Git\cmd;C:\Program Files\CMake\bin;C:\WINDOWS\system32;C:\WINDOWS;C:\WINDOWS\System32\Wbem;C:\WINDOWS\System32\WindowsPowerShell\v1.0\;C:\WINDOWS\System32\OpenSSH\;C:\Program Files\NVIDIA Corporation\NVIDIA NvDLISR"
SET "PYTHONPATH=%VIAME_BUILD_DIR%\install\lib\python3.6;%VIAME_BUILD_DIR%\install\lib\python3.6\site-packages"
SET "CUDA_BIN_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1"

git submodule update --init --recursive

"C:\Program Files\CMake\bin\ctest.exe" -S jenkins_dashboard.cmake -VV

REM ---------------------------------------------------
REM HACKS UNTIL THESE THINGS ARE BETTER HANDLED IN CODE
REM ---------------------------------------------------

SET GPU_ADD_ON_PACKAGE=C:\tmp\VIAME-Windows-GPU.zip
SET MISSING_SVM_DLL=%VIAME_SOURCE_DIR%\packages\smqtk\TPL\libsvm-3.1-custom\libsvm.dll
SET MISSING_DNET_EXE=%VIAME_BUILD_DIR%\build\src\darknet-build\Release\darknet.exe

MOVE "%VIAME_BUILD_DIR%\install" "%VIAME_BUILD_DIR%\VIAME"
"C:\Program Files\7-Zip\7z.exe" x -aoa %GPU_ADD_ON_PACKAGE% -o%VIAME_BUILD_DIR%
MOVE %MISSING_SVM_DLL% %VIAME_BUILD_DIR%\VIAME\bin
MOVE %MISSING_DNET_EXE% %VIAME_BUILD_DIR%\VIAME\bin

DEL "%VIAME_BUILD_DIR%\VIAME\lib\python3.6\site-packages\torch\lib\cu*"

COPY /Y %VIAME_SOURCE_DIR%\cmake\setup_viame.bat.install %VIAME_BUILD_DIR%\VIAME\setup_viame.bat

DIR /S /B "%VIAME_BUILD_DIR%\VIAME" > tmp.txt
TYPE tmp.txt | findstr /v "install\include" > files-core.lst

"C:\Program Files\7-Zip\7z.exe" a -tzip "%VIAME_BUILD_DIR%\VIAME-Core.zip" @files-core.lst

MOVE "%VIAME_BUILD_DIR%\VIAME" "%VIAME_BUILD_DIR%\VIAME-Core"

REM ---------------------------------------------------
REM Round2 - Build with torch
REM ---------------------------------------------------

REM DEL "%VIAME_BUILD_DIR%\CMakeCache.txt"
REM RD /S /Q "%VIAME_BUILD_DIR%\build\src\fletch-stamp"

XCOPY /E /I "%VIAME_BUILD_DIR%\VIAME-Core" "%VIAME_BUILD_DIR%\install"
DIR /S /B "%VIAME_BUILD_DIR%\install" > files-core.txt

git apply "%VIAME_SOURCE_DIR%\cmake\build_server_windows_msi-torch.diff"
COPY /Y "%VIAME_SOURCE_DIR%\cmake\build_server_windows_msi.cmake" platform.cmake
"C:\Program Files\CMake\bin\ctest.exe" -S jenkins_dashboard.cmake -VV

DIR /S /B "%VIAME_BUILD_DIR%\install" > tmp.txt
TYPE tmp.txt | findstr /v "VIAME\include" > files-torch.txt

FOR /f "delims=" %A in (files-torch.txt) do @find "%A" "files-core.txt" >nul2>nul || echo %A>>diff-torch.lst

"C:\Program Files\7-Zip\7z.exe" a -tzip "%VIAME_BUILD_DIR%\VIAME-Torch.zip" @diff-torch.lst

REM ---------------------------------------------------
REM Round3 - Build with darknet
REM ---------------------------------------------------

REM DEL "%VIAME_BUILD_DIR%\CMakeCache.txt"
REM RD /S /Q "%VIAME_BUILD_DIR%\build\src\fletch-stamp"

git apply "%VIAME_SOURCE_DIR%\cmake\build_server_windows_msi-darknet.diff"
COPY /Y "%VIAME_SOURCE_DIR%\cmake\build_server_windows_msi.cmake" platform.cmake
"C:\Program Files\CMake\bin\ctest.exe" -S jenkins_dashboard.cmake -VV

DIR /S /B "%VIAME_BUILD_DIR%\install" > tmp.txt
TYPE tmp.txt | findstr /v "install\include" > files-darknet.txt

FOR /f "delims=" %A in (files-darknet.txt) do @find "%A" "files-torch.txt" >nul2>nul || echo %A>>diff-darknet.lst

"C:\Program Files\7-Zip\7z.exe" a -tzip "%VIAME_BUILD_DIR%\VIAME-Darknet.zip" @diff-darknet.lst

REM ---------------------------------------------------
REM Round4 - Build with dive
REM ---------------------------------------------------

REM DEL "%VIAME_BUILD_DIR%\CMakeCache.txt"
REM RD /S /Q "%VIAME_BUILD_DIR%\build\src\fletch-stamp"

git apply "%VIAME_SOURCE_DIR%\cmake\build_server_windows_msi-dive.diff"
COPY /Y "%VIAME_SOURCE_DIR%\cmake\build_server_windows_msi.cmake" platform.cmake
"C:\Program Files\CMake\bin\ctest.exe" -S jenkins_dashboard.cmake -VV

DIR /S /B "%VIAME_BUILD_DIR%\install" > tmp.txt
TYPE tmp.txt | findstr /v "install\include" > files-dive.txt

FOR /f "delims=" %A in (files-dive.txt) do @find "%A" "files-darknet.txt" >nul2>nul || echo %A>>diff-dive.lst

"C:\Program Files\7-Zip\7z.exe" a -tzip "%VIAME_BUILD_DIR%\VIAME-DIVE.zip" @diff-dive.lst

REM ---------------------------------------------------
REM Round5 - Build with vivia
REM ---------------------------------------------------

REM DEL "%VIAME_BUILD_DIR%\CMakeCache.txt"
REM RD /S /Q "%VIAME_BUILD_DIR%\build\src\fletch-stamp"

git apply "%VIAME_SOURCE_DIR%\cmake\build_server_windows_msi-view.diff"
COPY /Y "%VIAME_SOURCE_DIR%\cmake\build_server_windows_msi.cmake" platform.cmake
"C:\Program Files\CMake\bin\ctest.exe" -S jenkins_dashboard.cmake -VV

DIR /S /B "%VIAME_BUILD_DIR%\install" > tmp.txt
TYPE tmp.txt | findstr /v "install\include" > files-view.txt

FOR /f "delims=" %A in (files-view.txt) do @find "%A" "files-dive.txt" >nul2>nul || echo %A>>diff-view.lst

"C:\Program Files\7-Zip\7z.exe" a -tzip "%VIAME_BUILD_DIR%\VIAME-VIEW.zip" @diff-view.lst

MOVE "%VIAME_BUILD_DIR%\install" "%VIAME_BUILD_DIR%\VIAME-VIEW"

REM ---------------------------------------------------
REM Round6 - Build with seal
REM ---------------------------------------------------

REM DEL "%VIAME_BUILD_DIR%\CMakeCache.txt"
REM RD /S /Q "%VIAME_BUILD_DIR%\build\src\fletch-stamp"

XCOPY /E /I "%VIAME_BUILD_DIR%\VIAME-Core" "%VIAME_BUILD_DIR%\install"
DIR /S /B "%VIAME_BUILD_DIR%\install" > files-core.txt

git reset --hard
git apply "%VIAME_SOURCE_DIR%\cmake\build_server_windows_msi-seal.diff"
COPY /Y "%VIAME_SOURCE_DIR%\cmake\build_server_windows_msi.cmake" platform.cmake
"C:\Program Files\CMake\bin\ctest.exe" -S jenkins_dashboard.cmake -VV

DIR /S /B "%VIAME_BUILD_DIR%\install" > tmp.txt
TYPE tmp.txt | findstr /v "install\include" > files-seal.txt

FOR /f "delims=" %A in (files-seal.txt) do @find "%A" "files-core.txt" >nul2>nul || echo %A>>diff-seal.lst

"C:\Program Files\7-Zip\7z.exe" a -tzip "%VIAME_BUILD_DIR%\VIAME-Torch.zip" @diff-seal.lst

MOVE "%VIAME_BUILD_DIR%\install" "%VIAME_BUILD_DIR%\VIAME-SEAL"
