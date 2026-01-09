@echo off
REM This file is part of VIAME, and is distributed under an OSI-approved
REM BSD 3-Clause License. See either the root top-level LICENSE file or
REM https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

REM Script to add or replace SSH remotes for git submodules in the packages directory.
REM
REM Usage:
REM   use_ssh_submodule_links.bat add      - Add SSH remote named "ssh" alongside origin
REM   use_ssh_submodule_links.bat replace  - Replace origin URL with SSH URL

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "PACKAGES_DIR=%SCRIPT_DIR%..\packages"

if "%~1"=="" (
    echo Usage: %~nx0 ^<add^|replace^>
    echo   add     - Add SSH remote named 'ssh' alongside existing origin
    echo   replace - Replace origin URL with SSH URL
    exit /b 1
)

set "MODE=%~1"

if /i not "%MODE%"=="add" if /i not "%MODE%"=="replace" (
    echo Error: Invalid mode '%MODE%'. Use 'add' or 'replace'.
    exit /b 1
)

echo Processing submodules in: %PACKAGES_DIR%
echo Mode: %MODE%
echo.

if not exist "%PACKAGES_DIR%" (
    echo Error: Packages directory not found: %PACKAGES_DIR%
    exit /b 1
)

REM Process each directory in packages
for /d %%D in ("%PACKAGES_DIR%\*") do (
    set "DIRNAME=%%~nxD"

    REM Skip downloads and patches directories
    if /i not "!DIRNAME!"=="downloads" if /i not "!DIRNAME!"=="patches" (
        call :process_repo "%%D"

        REM Also check subdirectories for nested submodules
        for /d %%S in ("%%D\*") do (
            call :process_repo "%%S"
        )
    )
)

echo.
echo Done!
exit /b 0

:process_repo
set "REPO_PATH=%~1"
set "REPO_NAME=%~nx1"

REM Check if it's a git repository
if not exist "%REPO_PATH%\.git" (
    if not exist "%REPO_PATH%\.git\HEAD" (
        exit /b 0
    )
)

pushd "%REPO_PATH%"

REM Get the origin URL
for /f "tokens=*" %%U in ('git remote get-url origin 2^>nul') do set "ORIGIN_URL=%%U"

if not defined ORIGIN_URL (
    echo   Skipping %REPO_NAME%: no origin remote
    popd
    exit /b 0
)

REM Check if it's an HTTPS URL
echo !ORIGIN_URL! | findstr /i "^https://" >nul
if errorlevel 1 (
    echo   Skipping %REPO_NAME%: origin is not HTTPS ^(!ORIGIN_URL!^)
    popd
    exit /b 0
)

REM Convert HTTPS to SSH URL
REM https://github.com/org/repo.git -> git@github.com:org/repo.git
set "SSH_URL=!ORIGIN_URL!"
set "SSH_URL=!SSH_URL:https://=git@!"
REM Replace first / after domain with :
for /f "tokens=1,* delims=/" %%A in ("!SSH_URL:git@=!") do (
    set "SSH_URL=git@%%A:/%%B"
)
set "SSH_URL=!SSH_URL:/=:!"
REM Fix double colons that might occur
set "SSH_URL=!SSH_URL:::=:!"

if /i "%MODE%"=="add" (
    REM Check if ssh remote already exists
    git remote get-url ssh >nul 2>&1
    if errorlevel 1 (
        echo   %REPO_NAME%: adding ssh remote -^> !SSH_URL!
        git remote add ssh "!SSH_URL!"
    ) else (
        echo   %REPO_NAME%: ssh remote already exists, updating URL
        git remote set-url ssh "!SSH_URL!"
    )
) else (
    echo   %REPO_NAME%: replacing origin -^> !SSH_URL!
    git remote set-url origin "!SSH_URL!"
)

popd
exit /b 0
