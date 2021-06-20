#!/bin/bash
for ((i=21;i<=33;i+=2)); do
	idx=$i
	file='submit_job_'$idx
	file=$file'.sh'
	echo $file
	source $file > exp.out
done
	# nohup mpirun -np $np python $RunScript > $ExpDir"/exp.out"
#nohup mpirun -np $np python $RunScript > ./exp.out




