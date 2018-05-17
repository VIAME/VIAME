call ..\..\..\setup_viame.bat

set DATABASE_DIR=database
set SQL_DIR=%DATABASE_DIR%\SQL
set INIT_FILE=configs\sql_init_table.sql
set LOG_FILE=database\SQL_Log_File

if "%1%"=="" (
  goto Usage
)

:: Batch doesn't let us use or, because of course not
if "%1%"=="init" (
  set 1=initialize
)

if "%1%"=="initialize" (
  pg_ctl stop -D %SQL_DIR%
  taskkill /F /IM postgres /T

  if exist "%DATABASE_DIR%\" rmdir /s /q %DATABASE_DIR%
  initdb -D %SQL_DIR%

  pg_ctl -w -t 20 -D %SQL_DIR% -l %LOG_FILE% start

  pg_ctl status -D %SQL_DIR%

  createuser -e -E -s -i -r -d postgres

  psql -f %INIT_FILE% postgres
  
  goto:eof
)

if "%1%"=="status" (
  pg_ctl status -D %SQL_DIR%
  
  goto:eof
)

if "%1%"=="start" (
  pg_ctl -w -t 20 -D %SQL_DIR% -l %LOG_FILE% start
  
  goto:eof
)

if "%1%"=="stop" (
  pg_ctl stop -D %SQL_DIR%
  pkill postgres
  
  goto:eof
)

:Usage
  echo "Usage: db_tool.bat [initialize | status | start | stop]"