#!/bin/bash

org_sub_file="submit_job_1.sh"
org_file="SE_valid_1.py"
echo $org_sub_file

for i in {2..6}
do
	j=$((i+16))
	l=$((j/10))
	p=$((j-l*10))
	new_sub_file="submit_job_$i.sh"
	echo $new_sub_file
	cp $org_sub_file $new_sub_file
	str="valid_$i"
	# str2="Exp$l_$p"
	str2="Exp$l"
	str2=$str2"_"
	str2=$str2"$p"
	sed -i "s/valid_1/$str/g" $new_sub_file
	sed -i "s/Exp1_7/$str2/g" $new_sub_file

	new_file="SE_valid_$i.py"
	cp $org_file $new_file
	sed -i "s/Exp1_7/$str2/g" $new_file
	str3="$j"
	sed -i "s/17/$str3/g" $new_file

done