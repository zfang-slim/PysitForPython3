#!/bin/bash
np=10
RunScript="Marmousi_1_Extra_3_0Hz.py"
ExpRootDir="/scratch/Zhilong/ExxonProject/MarmousiModel/Result/ExpPeriod201909/StartingFrequencyCompare/LSEX_Result"
ExpSubDir="/LS_Starting3_0HzExtraDownSample"
ExpDir=$ExpRootDir$ExpSubDir
WaveRootDir="/wavedata/Zhilong/ExxonProject/MarmousiModel/Result/ExpPeriod201909/StartingFrequencyCompare/LSEX_Result"
WaveDir=$WaveRootDir$ExpSubDir
ExpDir=$WaveRootDir$ExpSubDir
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




