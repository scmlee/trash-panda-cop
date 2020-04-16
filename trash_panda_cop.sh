#!/usr/bin/env bash
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 start|stop" >&2
  exit 1
fi

if [ "$1" = "start" ]; then
	python3 trash_panda_cop.py &
	if [ $# -eq 1 ]; then
		echo $! > /tmp/trash_panda_cop.pid
		echo "Trash panda cop started :)"
	else 
		echo "Trash panda cop not start. :("
	fi
elif [ "$1" = "stop" ]; then
	if ! [ -f "/tmp/trash_panda_cop.pid" ]; then
		echo "Could not find pid file." >&2
	else
		# TODO: Find a better way to kill the process; this doesn't allow the program to quit gracefully
		kill `cat /tmp/trash_panda_cop.pid` && echo "Stopped trash panda cop!"
	fi
else
	echo "Usage: $0 start|stop" >&2
	exit 1
fi
