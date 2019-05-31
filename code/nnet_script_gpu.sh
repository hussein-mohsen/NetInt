#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --job-name=tuning_experiment
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=25GB

start_time=$(date +%s)
date
module restore cuda
source activate dlnn

echo "$@"
date
python neuron_tuning.py "$@"
date

end_time=$(date +%s)

hours=$(( ($end_time-$start_time) / ( 3600 )))
minutes=$(( ($end_time-$start_time) / ( 60 )))
seconds=$(( ($end_time-$start_time) % 60 ))

echo "Execution time: " $hours "h" $(($minutes % (60))) "m" $seconds "s"
