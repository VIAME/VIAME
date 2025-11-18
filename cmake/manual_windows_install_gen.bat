REM -------------------------------------------------------------------------------------------------------
REM Setup Paths
REM -------------------------------------------------------------------------------------------------------

SET "BUILD_DIR=C:\VIAME-Builds\GPU"

IF EXIST %BUILD_DIR% rmdir /s /q %BUILD_DIR%

GIT clone https://github.com/VIAME/VIAME.git %BUILD_DIR%

CD %BUILD_DIR%

COPY %BUILD_DIR%\cmake\build_server_windows.bat build_server_windows.bat
COPY %BUILD_DIR%\cmake\build_server_windows.cmake platform.cmake
COPY %BUILD_DIR%\cmake\jenkins\CTestBuildOnlyPipeline jenkins_dashboard.cmake

CALL build_server_windows.bat > build_log.txt

PAUSE
