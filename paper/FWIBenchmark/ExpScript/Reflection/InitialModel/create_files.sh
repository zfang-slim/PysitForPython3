#!/bin/bash

org_sub_file="submit_job_0.sh"
org_file="RE_valid_PQN_0.py"
echo $org_sub_file

for i in {1..6}
do
	j=$((i+20))
	new_sub_file="submit_job_$i.sh"
	echo $new_sub_file
	cp $org_sub_file $new_sub_file
	str="PQN_$i"
	str2="Exp2_$i"
	sed -i "s/PQN_0/$str/g" $new_sub_file
	sed -i "s/Exp2_0/$str2/g" $new_sub_file

	new_file="RE_valid_PQN_$i.py"
	cp $org_file $new_file
	sed -i "s/Exp2_0/$str2/g" $new_file
	str3="=$j"
	echo $str3
	sed -i "s/=20/$str3/g" $new_file

done