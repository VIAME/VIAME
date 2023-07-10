REM -------------------------------------------------------------------------------------------------------
REM Setup Paths
REM -------------------------------------------------------------------------------------------------------

SET VIAME_SOURCE_DIR=C:\workspace\VIAME-Windows-GPU-MSI

SET VIAME_BUILD_DIR=%VIAME_SOURCE_DIR%\build
SET VIAME_INSTALL_DIR=%VIAME_BUILD_DIR%\install

REM Make sure to have all of these things installed (and cuDNN in CUDA)

SET "CMAKE_ROOT=C:\Program Files\CMake"
SET "GIT_ROOT=C:\Program Files\Git"
SET "ZIP_ROOT=C:\Program Files\7-Zip"
SET "ZLIB_ROOT=C:\Program Files\ZLib"
SET "NVIDIA_ROOT=C:\Program Files (x86)\NVIDIA Corporation"
SET "CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3"

SET "WIN_ROOT=C:\Windows"
SET "WIN32_ROOT=%WIN_ROOT%\System32"
SET "WIN64_ROOT=%WIN_ROOT%\SysWOW64"

REM Do not modify the below unless you are changing python versions or have alternatively modified
REM the build and install directories in cmake or the platforms.cmake file

SET "VIAME_BUILD_DIR=%VIAME_SOURCE_DIR%\build"
SET "VIAME_INSTALL_DIR=%VIAME_BUILD_DIR%\install"

SET "PYTHON_SUBDIR=lib\python3.6"

SET "PATH=%WIN_ROOT%;%WIN32_ROOT%;%WIN32_ROOT%\Wbem;%WIN32_ROOT%\WindowsPowerShell\v1.0;%WIN32_ROOT%\OpenSSH"
SET "PATH=%CUDA_ROOT%\bin;%CUDA_ROOT%\libnvvp;%NVIDIA_ROOT%\PhysX\Common;%NVIDIA_ROOT%\NVIDIA NvDLISR;%PATH%"
SET "PATH=%GIT_ROOT%\cmd;%CMAKE_ROOT%\bin;%PATH%"
SET "PYTHONPATH=%VIAME_INSTALL_DIR%\%PYTHON_SUBDIR%;%VIAME_INSTALL_DIR%\%PYTHON_SUBDIR%\site-packages"

REM -------------------------------------------------------------------------------------------------------
REM Round1 - VIAME Core
REM -------------------------------------------------------------------------------------------------------

IF EXIST build rmdir /s /q build

IF NOT EXIST C:\tmp mkdir C:\tmp

IF EXIST C:\tmp\fl5 rmdir /s /q C:\tmp\fl5
IF EXIST C:\tmp\kv5 rmdir /s /q C:\tmp\kv5
IF EXIST C:\tmp\vm5 rmdir /s /q C:\tmp\vm5

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
COPY %WIN64_ROOT%\vcomp140.dll %VIAME_INSTALL_DIR%\bin
COPY %WIN64_ROOT%\msvcr120.dll %VIAME_INSTALL_DIR%\bin
COPY "C:\Program Files\ZLib\dll_x64\zlibwapi.dll" %VIAME_INSTALL_DIR%\bin

powershell.exe "Get-ChildItem -Recurse "%VIAME_INSTALL_DIR%" | Resolve-Path -Relative" > tmp.txt
TYPE tmp.txt | findstr /v "install\include" > files-core.lst

"C:\Program Files\7-Zip\7z.exe" a -tzip "%VIAME_BUILD_DIR%\VIAME-Core.zip" @files-core.lst

MOVE "%VIAME_INSTALL_DIR%" "%VIAME_BUILD_DIR%\VIAME-Core"

REM -------------------------------------------------------------------------------------------------------
REM Round2 - Build with torch
REM -------------------------------------------------------------------------------------------------------

XCOPY /E /I "%VIAME_BUILD_DIR%\VIAME-Core" "%VIAME_INSTALL_DIR%"
powershell.exe "Get-ChildItem -Recurse %VIAME_INSTALL_DIR% | Resolve-Path -Relative" > files-core.txt

git apply "%VIAME_SOURCE_DIR%\cmake\build_server_windows_msi-torch.diff"
COPY /Y "%VIAME_SOURCE_DIR%\cmake\build_server_windows_msi.cmake" platform.cmake
"C:\Program Files\CMake\bin\ctest.exe" -S jenkins_dashboard.cmake -VV

DEL "%VIAME_INSTALL_DIR%\lib\python3.6\site-packages\torch\lib\cu*"

powershell.exe "Get-ChildItem -Recurse %VIAME_INSTALL_DIR% | Resolve-Path -Relative" > tmp.txt
TYPE tmp.txt | findstr /v "install\include" > files-torch.txt

FOR /f "delims=" %%A in (files-torch.txt) do @find "%%A" "files-core.txt" >nul2>nul || echo %%A>>diff-torch.lst

"C:\Program Files\7-Zip\7z.exe" a -tzip "%VIAME_BUILD_DIR%\VIAME-Torch.zip" @diff-torch.lst

REM -------------------------------------------------------------------------------------------------------
REM Round3 - Build with darknet
REM -------------------------------------------------------------------------------------------------------

git apply "%VIAME_SOURCE_DIR%\cmake\build_server_windows_msi-darknet.diff"
COPY /Y "%VIAME_SOURCE_DIR%\cmake\build_server_windows_msi.cmake" platform.cmake
"C:\Program Files\CMake\bin\ctest.exe" -S jenkins_dashboard.cmake -VV

COPY %VIAME_SOURCE_DIR%\packages\darknet\3rdparty\pthreads\bin\pthreadVC2.dll %VIAME_INSTALL_DIR%\bin

powershell.exe "Get-ChildItem -Recurse %VIAME_INSTALL_DIR% | Resolve-Path -Relative" > tmp.txt
TYPE tmp.txt | findstr /v "install\include" > files-darknet.txt

FOR /f "delims=" %%A in (files-darknet.txt) do @find "%%A" "files-torch.txt" >nul2>nul || echo %%A>>diff-darknet.lst

"C:\Program Files\7-Zip\7z.exe" a -tzip "%VIAME_BUILD_DIR%\VIAME-Darknet.zip" @diff-darknet.lst

REM -------------------------------------------------------------------------------------------------------
REM Round4 - Build with dive
REM --------------------------------------------------------------------------------------------------------

git apply "%VIAME_SOURCE_DIR%\cmake\build_server_windows_msi-dive.diff"
COPY /Y "%VIAME_SOURCE_DIR%\cmake\build_server_windows_msi.cmake" platform.cmake
"C:\Program Files\CMake\bin\ctest.exe" -S jenkins_dashboard.cmake -VV

powershell.exe "Get-ChildItem -Recurse %VIAME_INSTALL_DIR% | Resolve-Path -Relative" > tmp.txt
TYPE tmp.txt | findstr /v "install\include" > files-dive.txt

FOR /f "delims=" %%A in (files-dive.txt) do @find "%%A" "files-darknet.txt" >nul2>nul || echo %%A>>diff-dive.lst

"C:\Program Files\7-Zip\7z.exe" a -tzip "%VIAME_BUILD_DIR%\VIAME-DIVE.zip" @diff-dive.lst

REM -------------------------------------------------------------------------------------------------------
REM Round5 - Build with vivia
REM -------------------------------------------------------------------------------------------------------

git apply "%VIAME_SOURCE_DIR%\cmake\build_server_windows_msi-view.diff"
COPY /Y "%VIAME_SOURCE_DIR%\cmake\build_server_windows_msi.cmake" platform.cmake
"C:\Program Files\CMake\bin\ctest.exe" -S jenkins_dashboard.cmake -VV

powershell.exe "Get-ChildItem -Recurse %VIAME_INSTALL_DIR% | Resolve-Path -Relative" > tmp.txt
TYPE tmp.txt | findstr /v "install\include" > files-view.txt

FOR /f "delims=" %%A in (files-view.txt) do @find "%%A" "files-dive.txt" >nul2>nul || echo %%A>>diff-view.lst

"C:\Program Files\7-Zip\7z.exe" a -tzip "%VIAME_BUILD_DIR%\VIAME-VIEW.zip" @diff-view.lst

MOVE "%VIAME_INSTALL_DIR%" "%VIAME_BUILD_DIR%\VIAME-VIEW"

REM -------------------------------------------------------------------------------------------------------
REM Round6 - Build with seal
REM -------------------------------------------------------------------------------------------------------

XCOPY /E /I "%VIAME_BUILD_DIR%\VIAME-Core" "%VIAME_INSTALL_DIR%"
powershell.exe "Get-ChildItem -Recurse %VIAME_INSTALL_DIR% | Resolve-Path -Relative" > files-core.txt

git reset --hard
git apply "%VIAME_SOURCE_DIR%\cmake\build_server_windows_msi-seal.diff"
COPY /Y "%VIAME_SOURCE_DIR%\cmake\build_server_windows_msi.cmake" platform.cmake
"C:\Program Files\CMake\bin\ctest.exe" -S jenkins_dashboard.cmake -VV

powershell.exe "Get-ChildItem -Recurse %VIAME_INSTALL_DIR% | Resolve-Path -Relative" > tmp.txt
TYPE tmp.txt | findstr /v "install\include" > files-seal.txt

FOR /f "delims=" %%A in (files-seal.txt) do @find "%%A" "files-core.txt" >nul2>nul || echo %%A>>diff-seal.lst

"C:\Program Files\7-Zip\7z.exe" a -tzip "%VIAME_BUILD_DIR%\VIAME-Torch.zip" @diff-seal.lst

MOVE "%VIAME_INSTALL_DIR%" "%VIAME_BUILD_DIR%\VIAME-SEAL"
