#!/bin/bash

org_sub_file="submit_job_1.sh"
org_file="Marmousi_1.py"
echo $org_sub_file

for i in {2..5}
do
	j=$((i+24))
	l=$((j/10))
	p=$((j-l*10))
	new_sub_file="submit_job_$i.sh"
	echo $new_sub_file
	cp $org_sub_file $new_sub_file
	str="Marmousi_$i"
	# str2="Exp$l_$p"
	str2="Starting$l"
	str2=$str2"_"
	str2=$str2"$p"
	sed -i "s/Marmousi_1/$str/g" $new_sub_file
	sed -i "s/Starting2_5/$str2/g" $new_sub_file

	new_file="Marmousi_$i.py"
	cp $org_file $new_file
	sed -i "s/Starting2_5/$str2/g" $new_file
	str3="$l"
	str3=$str3"_"
	str3=$str3"$p"
	str3=$str3"Hz"
	sed -i "s/2_5Hz/$str3/g" $new_file

done