#!/bin/bash

source ~/Dev/viame/build/install/setup_viame.sh

DATABASE_DIR=database
SQL_DIR=${DATABASE_DIR}/SQL
INIT_FILE=configs/sql_init_table.sql
LOG_FILE=database/SQL_Log_File

if [ "$1" == "initialize" ] || [ "$1" == "init" ] ;
then

  pg_ctl stop -D ${SQL_DIR}
  pkill postgres

  rm -rf ${DATABASE_DIR}
  initdb -D ${SQL_DIR}

  pg_ctl -w -t 20 -D ${SQL_DIR} -l ${LOG_FILE} start

  pg_ctl status -D ${SQL_DIR}
  createuser -e -E -s -i -r -d postgres

  psql -f ${INIT_FILE} postgres
elif [ "$1" == "status" ]
then
  pg_ctl status -D ${SQL_DIR}

elif [ "$1" == "start" ]
then
  pg_ctl -w -t 20 -D ${SQL_DIR} -l ${LOG_FILE} start

elif [ "$1" == "stop" ]
then
  pg_ctl stop -D ${SQL_DIR}
  pkill postgres

else
  echo "Usage: db_tool.sh [initialize | status | start | stop]"
fi
