#!/usr/bin/env bash
#
# Start mongo database some where with:
#	mongod --dbpath $PWD
# Where $PWD is some directory
#
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../../.."

source ${VIAME_INSTALL}/setup_viame.sh 

# Start the process, storing the pid
# Let it run in the background, keeping a log
SMQTK_GUI_IQR_PID="smqtk_iqr.pid"
echo "Starting SMQTK IqrSearchDispatcher..."
runApplication -a IqrSearchDispatcher \
  -c configs/run_app.iqr_dispatcher.json \
  -tv \
  &>"./runApp.IqrSearchDispatcher.log" &
echo "$!" >"${SMQTK_GUI_IQR_PID}"
echo "Starting SMQTK IqrSearchDispatcher... Done"

# Kill the IqrSearchDispatcher and wait for it to end
function process_cleanup() {
  signal="$1"

  echo "Stopping IQR GUI app"
  kill -${signal} $(cat "${SMQTK_GUI_IQR_PID}")

  wait $(cat "${SMQTK_GUI_IQR_PID}")
  rm "${SMQTK_GUI_IQR_PID}"
}

# Call the process to kill IqrSearchDispatcher when this process is killed
trap "process_cleanup SIGTERM;" HUP INT TERM
trap "process_cleanup SIGKILL;" QUIT KILL

# Sit back and wait for the user to be done
wait $(cat "${SMQTK_GUI_IQR_PID}")
