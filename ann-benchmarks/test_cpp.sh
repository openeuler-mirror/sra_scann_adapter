#!/bin/bash
set -x
 
# 根据自己的项目路径进行修改
project_path=$(cd $(dirname $0)/../; pwd)
echo "project_path=" $project_path
 
GCC=gcc
 
 
function ann_b_batch_expand_multiprocess() {
  
  for ((i=1; i<=$1; i++)); do
    #for data in glove-100-angular sift-128-euclidean deep-image-96-angular fashion-mnist-784-euclidean gist-960-euclidean ; do
    for data in sift-128-euclidean; do
      echo ""
      echo "============ test batch times ${i} ================"
      echo ""
 
      cd ${project_path}/scann
      dataDir=${project_path}/ann-benchmarks/data/${data}.hdf5
      configDir=${project_path}/ann-benchmarks/ann_benchmarks/algorithms/scann/cpp_test/config-${data}.config
 
      time numactl -N 0 -m 0 -- ./bazel_bin/eval --dataDir=$dataDir --configDir=$configDir --numRuns=8&
      #time numactl -N 1 -m 1 -- ./bazel_bin/eval --dataDir=$dataDir --configDir=$configDir --numRuns=8&
      #time numactl -N 2 -m 2 -- ./bazel_bin/eval --dataDir=$dataDir --configDir=$configDir --numRuns=8&
      #time numactl -N 3 -m 3 -- ./bazel_bin/eval --dataDir=$dataDir --configDir=$configDir --numRuns=8&
 
      wait
    done
  done
}
function main() {
  
  for i in {1..1}; do
    
    ann_b_batch_expand_multiprocess 1 # 设置batch执行次数
    
  done
}
 
main