#!/bin/bash
for ((i=3;i<=9;i+=2)); do
	idx=$i
	file='submit_job_PQN'$idx
	file=$file'.sh'
	echo $file
	source $file > exp.out
done
	# nohup mpirun -np $np python $RunScript > $ExpDir"/exp.out"
#nohup mpirun -np $np python $RunScript > ./exp.out




