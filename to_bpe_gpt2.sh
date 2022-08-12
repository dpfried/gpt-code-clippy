#!/bin/bash

for input_dir in \
  /checkpoint/dpf/data/bigquery/jupyter/data_dedup_filtered_mwcf-0.4_mll-3000_pandoc_csn-conservative
do
  dn="`dirname $input_dir`"
  name="gpt2-`basename $dn`"
  sbatch -J $name -o 'slurm-logs/slurm-%j-%x.out' -e 'slurm-logs/slurm-%j-%x.err' to_bpe_gpt2_single.sh $input_dir
done


  # /checkpoint/dpf/data/google-code/python_open-source/data_dedup_filtered_mwcf-0.4_mll-3000_pandoc_csn-conservative\
  # /checkpoint/dpf/data/google-code/javascript_open-source/data_dedup_filtered_mwcf-0.4_mll-3000_pandoc_csn-conservative\
  # /checkpoint/dpf/data/bitbucket/python_open-source/data_dedup_filtered_mwcf-0.4_mll-3000_pandoc_csn-conservative\
  # /checkpoint/dpf/data/bitbucket/javascript_open-source/data_dedup_filtered_mwcf-0.4_mll-3000_pandoc_csn-conservative

  # /checkpoint/dpf/data/github/javascript_forkless_open-source/data_dedup_filtered_mwcf-0.4_mll-3000_pandoc_csn-conservative\
  # /checkpoint/dpf/data/github/python_forkless_open-source_2star+/data_dedup_filtered_mwcf-0.4_mll-3000_pandoc_csn-conservative\
  # /checkpoint/dpf/data/github/jupyter_forkless_open-source_2021_1star+/data_dedup_filtered_mwcf-0.4_mll-3000_pandoc_csn-conservative\
  # /checkpoint/dpf/data/github/python_forkless_open-source_0stars_20K+/data_dedup_filtered_mwcf-0.4_mll-3000_pandoc_csn-conservative\
  # /checkpoint/dpf/data/github/python_forkless_open-source_0stars_20K+_2020/data_dedup_filtered_mwcf-0.4_mll-3000_pandoc_csn-conservative\
  # /checkpoint/dpf/data/github/python_forkless_open-source_1star/data_dedup_filtered_mwcf-0.4_mll-3000_pandoc_csn-conservative\
  # /checkpoint/dpf/data/gitlab/python_open-source/data_dedup_filtered_mwcf-0.4_mll-3000_pandoc_csn-conservative\
  # /checkpoint/dpf/data/gitlab/javascript_open-source/data_dedup_filtered_mwcf-0.4_mll-3000_pandoc_csn-conservative\
  # /checkpoint/dpf/data/github/jupyter_forkless_open-source_2021_0stars_20K+/data_dedup_filtered_mwcf-0.4_mll-3000_pandoc_csn-conservative\
  # /checkpoint/dpf/data/github/jupyter_forkless_open-source_to2020_1star+/data_dedup_filtered_mwcf-0.4_mll-3000_pandoc_csn-conservative\
  # /checkpoint/dpf/data/bigquery/jupyter/data_dedup_filtered_mwcf-0.4_mll-3000_pandoc_csn-conservative\
  # /checkpoint/dpf/data/bigquery/python_exclude_12-15_no-forks/data_dedup_filtered_mwcf-0.4_mll-3000_pandoc_csn-conservative
