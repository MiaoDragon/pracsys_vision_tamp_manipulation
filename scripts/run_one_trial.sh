#!/usr/bin/env bash
roslaunch motoman_moveit_config move_group.launch > moveit_log.txt 2>&1 &

num=6
t=1
TRIAL=trial_$t
if [ -f $TRIAL.pkl ]
then
	echo "Loading and starting $TRIAL with $num objects."
	python sim_scene.py 1 $TRIAL y > log_exec.txt 2>&1 &
else
	echo "Generating and starting $TRIAL with $num objects."
	echo -e "0\n$num\n1\n$TRIAL\n" | python sim_scene.py 0 $TRIAL y  > log_exec.txt 2>&1 &
fi
python task_planner_retrieval.py $TRIAL > log_plan.txt 2>&1
echo "Finished $TRIAL"
pkill -9 -f sim_scene
pkill -9 -f ros
# pkill -9 -f sim_scene
# pkill -9 -f task_planner