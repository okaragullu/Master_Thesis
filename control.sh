#!/usr/local_rwth/bin/zsh
### Job name 
#BATCH --job-name=Automate
### File/Path where STDOUT will be written to, %J is the job id
#SBATCH --output ZControl_out.%J
### Request the time you need for execution. The full format is D-HH:MM:SS
### You must at least specify minutes or days and hours and may add or
### leave out any other parameters
#SBATCH --time=120:00:00
### Apply for Project
#SBATCH --mem-per-cpu=50M
#SBATCH --ntasks=1


dst="/rwthfs/rz/cluster/home/ro952148/Sim_Fin"
src="/rwthfs/rz/cluster/home/ro952148/Simulation"
sim="/rwthfs/rz/cluster/home/ro952148/Scripts"
cd $src
dirarray=()
###directory array
let num_sim=5
###max amount of scripts being run

listing () {
cd $src
dirarray=(./*)
}
###function for creating directory array

listing
while true
do
while [[ ${#dirarray[@]} -gt 0 ]] && [[ $(squeue -u ro952148|wc -l) -lt $num_sim ]];do
listing
for (( x=1; x<=${#dirarray[@]}; x++ ));do
cd $src
cd $dirarray[x]
elif [[ $(ls|wc -l)>3 ]] && [[ -f $(find -name *out) ]];then
mv -i $(pwd) $dst
elif [[ $(ls|wc -l) -eq 3 ]];then
sbatch batch.sh
while [[ $(ls|wc -l) -eq 3]];do
sleep 30
done
cd $src
fi
done
done
sleep 5
while [[ ${#dirarray[@]} -gt 0 ]] && [[ $(squeue -u ro952148|wc -l) -ge $num_sim ]];do
listing
for (( x=1; x<=$(ls|wc -l); x++ ));do 
cd $dirarray[x]
if [[ $(ls|wc -l)>3]] && [[ -f $(find -name *out) ]];then
mv -i $(pwd) $dst
else
cd $src
fi
done
done
done