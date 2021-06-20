#!/bin/bash

org_sub_file="submit_job_1.sh"
org_file="Marmousi_1.py"
echo $org_sub_file

for i in {2..16}
do
	j=$((i+4))
	new_sub_file="submit_job_$i.sh"
	echo $new_sub_file
	cp $org_sub_file $new_sub_file
	str="Marmousi_$i"
	str2="Initial$j"
	sed -i "s/Marmousi_1/$str/g" $new_sub_file
	sed -i "s/Initial5/$str2/g" $new_sub_file

	new_file="Marmousi_$i.py"
	echo $new_file
	cp $org_file $new_file
	sed -i "s/Initial5/$str2/g" $new_file
	str3="gskw=$j"
	echo $str3
	sed -i "s/gskw=5/$str3/g" $new_file

done


