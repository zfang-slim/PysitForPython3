#!/bin/bash
for ((i=2;i<=10;i+=1)); do
	idx=$i
	file='submit_job_'$idx
	file=$file'.sh'
	echo $file
	source $file > exp.out
done
	# nohup mpirun -np $np python $RunScript > $ExpDir"/exp.out"
#nohup mpirun -np $np python $RunScript > ./exp.out




