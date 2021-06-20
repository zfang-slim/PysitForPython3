#!/bin/bash

org_sub_file="submit_job_20.sh"
org_file="SF_valid_20.py"
echo $org_sub_file

for i in {21..22}
do
	new_sub_file="submit_job_$i.sh"
	echo $new_sub_file
	cp $org_sub_file $new_sub_file
	str="valid_$i"
	str2="Exp2_$i"
	sed -i "s/valid_20/$str/g" $new_sub_file
	sed -i "s/Exp2_20/$str2/g" $new_sub_file

	new_file="SF_valid_$i.py"
	cp $org_file $new_file
	sed -i "s/Exp2_20/$str2/g" $new_file
	str3="2.$i"
	echo $str3
	sed -i "s/2.20/$str3/g" $new_file

done