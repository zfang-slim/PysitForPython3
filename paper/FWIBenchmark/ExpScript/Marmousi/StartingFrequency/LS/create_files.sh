#!/bin/bash

org_sub_file="submit_job_1.sh"
org_file="Marmousi_1.py"
echo $org_sub_file

for i in {2..15}
do
	j=$((i+20))
	l=$((j/10))
	p=$((j-l*10))
	q=$((l+2))
	s=$((j-5))
	new_sub_file="submit_job_$i.sh"
	echo $new_sub_file
	cp $org_sub_file $new_sub_file
	str="Marmousi_$i"
	# str2="Exp$l_$p"
	str2="Starting$l"
	str2=$str2"_"
	str2=$str2"$p"
	str2=$str2"_"
	str2=$str2"$q"
	str2=$str2"_"
	str2=$str2"$p"
	sed -i "s/Marmousi_1/$str/g" $new_sub_file
	sed -i "s/Starting2_1_4_1/$str2/g" $new_sub_file

	new_file="Marmousi_$i.py"
	cp $org_file $new_file
	sed -i "s/Starting2_1_4_1/$str2/g" $new_file
	str3="=$s"
	sed -i "s/=16/$str3/g" $new_file

done