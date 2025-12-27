REM -------------------------------------------------------------------------------------------------------
REM Setup Paths
REM -------------------------------------------------------------------------------------------------------

SET "BUILD_DIR=C:\VIAME-Builds\GPU"

IF EXIST %BUILD_DIR% rmdir /s /q %BUILD_DIR%

git clone https://github.com/VIAME/VIAME.git %BUILD_DIR%

CD %BUILD_DIR%

REM The build script auto-generates the ctest dashboard file
CALL cmake\build_server_windows.bat > build_log.txt

PAUSE
