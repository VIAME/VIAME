#!/bin/bash

#+
# Allows simple testing of ZeroMQ pipelines.  Designed to test 1 Reciever and
# two senders.  Usage:
#
#     ./test_zmq_with_tmux.sh  <environment_setup_script>
#
# Where environment_setup_script is a scripe (possibley setup_kiwver.sh) that
# sets the environent up properly so that pipleine_runner will be found in
# PATH and will execute properly
#
# TMUX session will create 2 panes (top) with senders, and a third pane
# (bottom) that will prepare to recieve the 2 senders.  Simply  attach
# to the session, select the bottom pane,  and hit return to start
# the test.
#-

SESSION="kwiver_sprokit_zeromq"
START_SCRIPT=$1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd ${SCRIPT_DIR}

echo "$(date) Creating session ${SESSION}"
tmux new-session -d -s $SESSION

sleep 1
echo "$(date) Starting Sender..."
tmux select-window -t $SESSION:0
tmux rename-window -t $SESSION:0 'Sender0'
tmux send-keys -t $SESSION:0 "cd ${SCRIPT_DIR}" C-m
tmux send-keys -t $SESSION:0 "source ${START_SCRIPT}" C-m
tmux send-keys -t $SESSION:0 "kwiver runner ${SCRIPT_DIR}/test_zmq_send.pipe --setting sim:reader:simulator:detection_class=simulated --setting zmq:expected_subscribers=2" C-m

sleep 1
tmux split-window -t $SESSION:0
#tmux select-pane -t 0
tmux split-window -h -t $SESSION:0

sleep 1
echo "$(date) First Receiver..."
tmux select-pane -t 1
tmux send-keys -t 1 "cd ${SCRIPT_DIR}" C-m
tmux send-keys -t 1 "source ${START_SCRIPT}" C-m
tmux send-keys -t 1 "kwiver runner ${SCRIPT_DIR}/test_zmq_recv.pipe --setting sink::file_name=received_dos_one.csv" C-m

sleep 1
echo "$(date) Preparing Second Receiver..."
tmux send-keys -t 2 "cd ${SCRIPT_DIR}" C-m
tmux send-keys -t 2 "source ${START_SCRIPT}" C-m
tmux send-keys -t 2 "kwiver runner  ${SCRIPT_DIR}/test_zmq_recv.pipe --setting sink::file_name=received_dos_two.csv"

echo "$(date) Test prepared!"
