#!/bin/bash
np=20
RunScript="Marmousi_1_2.py"
ExpRootDir="/scratch/Zhilong/ExxonProject/MarmousiModel/Result/ExpPeriod201909_2/InitialModel"
ExpSubDir="/LS_Compress_Result/LS_BoxConstraint_SmoothInitial500_CorrectBox2"
ExpDir=$ExpRootDir$ExpSubDir
WaveRootDir="/wavedata/Zhilong/ExxonProject/MarmousiModel/Result/ExpPeriod201909_2/InitialModel"
WaveDir=$WaveRootDir$ExpSubDir
ExpDir=$WaveDir
echo $np 
echo $RunScript
echo $ExpDir"/exp.out"
echo $WaveDir

if  [ ! -d $ExpDir ]; then 
	mkdir -p $ExpDir
fi

if  [ ! -d $WaveDir ]; then
mkdir -p $WaveDir
fi

job_history_file='/math/home/fangzl/job_history.txt'
a=$(grep -o -c INDEX $job_history_file)
b=$((a+1))
expind='EXP INDEX     : '$b
expindcmd='1i\'$expind
sed -i "$expindcmd" $job_history_file

ScriptPath=$(pwd)'/'$RunScript
ScriptPathcmd='2i\Script Path   : '$ScriptPath
sed -i "$ScriptPathcmd" $job_history_file

ExpDircmd='3i\Exp Dir       : '$ExpDir
sed -i "$ExpDircmd" $job_history_file

WaveDircmd='4i\Wave Dir      : '$WaveDir
sed -i "$WaveDircmd" $job_history_file


host_name=$(uname -n)
hostcmd='5i\Host          : '$host_name
sed -i "$hostcmd" $job_history_file

time=$(date)
timecmd='6i\Submitted Time: '$time
sed -i "$timecmd" $job_history_file


sed -i '7i\ ' $job_history_file
sed -i '8i\ ' $job_history_file
sed -i '9i\ ' $job_history_file


nohup mpirun -np $np python $RunScript > $ExpDir"/exp.out"
#nohup mpirun -np $np python $RunScript > ./exp.out




