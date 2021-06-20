#!/bin/bash
np=10
RunScript="RE_valid_PQN_OTQadd.py"
ExpRootDir="/wavedata/Zhilong/ExxonProject/ReflectModel/ExxonMeeting201909/Reflector_PQN2_Test/Exp2_0"
ExpSubDir="/"
ExpDir=$ExpRootDir$ExpSubDir
WaveRootDir="/wavedata/Zhilong/ExxonProject/ReflectModel/ExxonMeeting201909/Reflector_PQN2_Test/Exp2_0"
WaveDir=$WaveRootDir$ExpSubDir
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




