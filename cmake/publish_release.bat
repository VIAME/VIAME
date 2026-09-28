@ECHO OFF
SETLOCAL EnableDelayedExpansion

REM ---------------------------------------------------------------------------
REM Publish a built package to its permanent Girder and Google Drive locations
REM and refresh the version text in README.md. Both mirrors keep one file whose
REM contents are replaced, so the README download links never change.
REM
REM Deliberately NOT called by build_server_windows.bat on its own: publishing
REM only happens when VIAME_PUBLISH=1 is set, so a developer build can never
REM upload by accident.
REM
REM Required environment:
REM   VIAME_PUBLISH=1     opt in
REM   GIRDER_API_KEY      data.kitware.com key with write access to the folder
REM   RCLONE_EXE          path to rclone.exe (default: rclone on PATH)
REM   RCLONE_REMOTE       rclone remote for the shared drive (default: viame-drive)
REM
REM Usage: publish_release.bat <package> [platform] [flavor]
REM ---------------------------------------------------------------------------

SET "PACKAGE=%~1"
SET "PLATFORM=%~2"
SET "FLAVOR=%~3"
IF "%PLATFORM%"=="" SET "PLATFORM=windows"
IF "%FLAVOR%"=="" SET "FLAVOR=gpu"

IF "%PACKAGE%"=="" (
    ECHO [publish] ERROR: no package given
    EXIT /B 1
)
IF NOT EXIST "%PACKAGE%" (
    ECHO [publish] ERROR: package not found: %PACKAGE%
    EXIT /B 1
)

IF NOT "%VIAME_PUBLISH%"=="1" (
    ECHO [publish] VIAME_PUBLISH is not 1; skipping upload of %PACKAGE%
    EXIT /B 0
)

REM A broken build must never be published.
ECHO %PACKAGE% | FINDSTR /I "BROKEN" >NUL
IF NOT ERRORLEVEL 1 (
    ECHO [publish] ERROR: refusing to publish %PACKAGE%
    EXIT /B 1
)

SET "VIAME_SOURCE_DIR=%~dp0.."
SET "VIAME_PYTHON=%VIAME_SOURCE_DIR%\build\install\bin\python.exe"
IF NOT EXIST "%VIAME_PYTHON%" SET "VIAME_PYTHON=python"

ECHO [publish] %DATE% %TIME% publishing %PACKAGE%

"%VIAME_PYTHON%" "%~dp0publish_release.py" "%PACKAGE%" ^
    --platform %PLATFORM% --flavor %FLAVOR% ^
    --readme "%VIAME_SOURCE_DIR%\README.md" %VIAME_PUBLISH_EXTRA_ARGS%
IF ERRORLEVEL 1 (
    ECHO [publish] ERROR: publishing failed
    EXIT /B 1
)

ECHO [publish] %DATE% %TIME% complete
EXIT /B 0
