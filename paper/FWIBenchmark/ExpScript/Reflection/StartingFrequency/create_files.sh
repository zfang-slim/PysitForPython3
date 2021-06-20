#!/bin/bash

org_sub_file="submit_job_PQN1.sh"
org_file="RE_valid_PQN_1.py"
echo $org_sub_file


for i in {2..50}
do
	j=$((i+9))
	l=$((j/10))
	p=$((j-l*10))
	new_sub_file="submit_job_PQN$i.sh"
	echo $new_sub_file
	cp $org_sub_file $new_sub_file
	str="valid_PQN_$i"
	str2="Exp$l"
	str2=$str2"_"
	str2=$str2"$p"
	sed -i "s/valid_PQN_1/$str/g" $new_sub_file
	sed -i "s/Exp1_0/$str2/g" $new_sub_file

	new_file="RE_valid_PQN_$i.py"
	cp $org_file $new_file
	sed -i "s/Exp1_0/$str2/g" $new_file
	str3="sfreq=$j"
	echo $str3
	sed -i "s/sfreq=10/$str3/g" $new_file

done