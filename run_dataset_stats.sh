#!/bin/bash

#SBATCH --time=4320
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --partition=devlab
#SBATCH --mem-per-cpu=2G

#data_dir=$1
# source=$2

# if [ -z $source ]
# then
#   source="github"
# fi

# python -u dataset_stats_par.py $data_dir \
#   --source $source \
#   --tokenizer_names gpt2 \
#   --n_procs 20 \
#   | tee ${data_dir}/size_stats.out

# python -u dataset_stats_par.py $data_dir \
#   --source $source \
#   --tokenizer_names bpe bpe_rn \
#   --n_procs 20 \
#   | tee ${data_dir}/size_stats_bpe.out

for data_dir in $@
do
  python -u dataset_stats_par.py $data_dir \
    --tokenizer_names gpt2 codet5 ours \
    --n_procs 40 \
    | tee ${data_dir}/size_stats_jupyter_proc.out
done

  #--source $source \
