@ECHO OFF
SETLOCAL EnableDelayedExpansion

REM ============================================================================
REM VIAME Build Monitor
REM Runs in background and periodically reports build progress to console.
REM
REM Arguments:
REM   %1 = Path to build_log.txt
REM
REM Stops when build_monitor_stop.flag appears in the build directory.
REM ============================================================================

SET "LOG_FILE=%~1"
SET "BUILD_DIR=%~dp1"
SET "STOP_FLAG=%BUILD_DIR%build_monitor_stop.flag"
SET "LAST_SIZE=0"
SET "CHECK_INTERVAL=30"

REM Remove stale stop flag
IF EXIST "%STOP_FLAG%" DEL "%STOP_FLAG%"

TITLE VIAME Build Monitor

:MONITOR_LOOP
    REM Check for stop signal
    IF EXIST "%STOP_FLAG%" (
        ECHO [Monitor] Build finished, stopping monitor.
        DEL "%STOP_FLAG%" 2>NUL
        GOTO :MONITOR_END
    )

    REM Check if the log file exists yet
    IF NOT EXIST "%LOG_FILE%" (
        PING -n %CHECK_INTERVAL% 127.0.0.1 >NUL
        GOTO :MONITOR_LOOP
    )

    REM Report on build progress by checking for key milestones in the log
    REM Look for completed ExternalProject builds (the main progress indicator)
    SET "COMPLETED_PROJECTS="
    SET "PROJ_COUNT=0"
    FOR /F "tokens=*" %%L IN ('FINDSTR /C:"Completed " "%LOG_FILE%" 2^>NUL') DO (
        SET "COMPLETED_PROJECTS=%%L"
        SET /A PROJ_COUNT+=1
    )

    REM Look for any errors
    SET "ERROR_COUNT=0"
    FOR /F %%N IN ('FINDSTR /C:"FAILED" /C:"Error" /C:"error C" "%LOG_FILE%" 2^>NUL ^| FIND /C /V ""') DO (
        SET "ERROR_COUNT=%%N"
    )

    REM Show current status
    ECHO [%TIME%] Projects completed: !PROJ_COUNT! ^| Errors/warnings: !ERROR_COUNT!
    IF DEFINED COMPLETED_PROJECTS (
        ECHO            Latest: !COMPLETED_PROJECTS!
    )

    REM Wait before checking again
    PING -n %CHECK_INTERVAL% 127.0.0.1 >NUL
    GOTO :MONITOR_LOOP

:MONITOR_END
ENDLOCAL
EXIT /B 0
