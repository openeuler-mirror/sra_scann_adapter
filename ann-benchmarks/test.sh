#!/bin/bash
set -x

# 根据自己的项目路径进行修改
project_path=$(cd $(dirname $0)/../; pwd)
echo "project_path=" $project_path

GCC=gcc

function git_info() {
  git branch
  git log -n 1
  git status
}

function expand_query() {
  # 扩大query倍数
  sed -n '122p' ./ann_benchmarks/runner.py
  sed -i '122s/.*/            results = batch_query(X_test, '$1')  # expanded n times/' ./ann_benchmarks/runner.py
  sed -n '122p' ./ann_benchmarks/runner.py
}

function ann_b_batch_expand_multiprocess() {
  cd ${project_path}/ann-benchmarks
    git_info

  expand_query 100

  cd ${project_path}/ann-benchmarks/ann_benchmarks/algorithms/scann
  if [ ! -f module.py.bak ]; then
    cp module.py module.py.bak
  fi
  cd ${project_path}/ann-benchmarks

  for ((i=1; i<=$1; i++)); do
      #for data in glove-100-angular deep-image-96-angular fashion-mnist-784-euclidean sift-128-euclidean gist-960-euclidean; do
      for data in sift-128-euclidean; do
        echo ""
        echo "============ test batch times ${i} ================"
        echo ""

        cd ${project_path}/ann-benchmarks/ann_benchmarks/algorithms/scann
        cp config-${data}.yml config.yml

        cp module.py.bak module.py

        case $data in
          "glove-100-angular")
            sed -i '28s/\(\.tree(self\.n_leaves, 1, training_sample_size=len(X), spherical=spherical, quantize_centroids=True\))/\1, soar_lambda=0.5, overretrieve_factor=1, distance_measure="dot_product")/' module.py
            ;;
          "deep-image-96-angular")
            ssed -i '28s/\(\.tree(self\.n_leaves, 1, training_sample_size=len(X), spherical=spherical, quantize_centroids=True\))/\1, soar_lambda=0.5, overretrieve_factor=1.6, distance_measure="dot_product")/' module.py
            ;;
          *)
            ;;
        esac

        rm -rf ${project_path}/ann-benchmarks/results/${data}/10/scann-batch
        cd ${project_path}/ann-benchmarks

        time numactl -N 0 -m 0 -- python run.py --dataset ${data} --algorithm scann --local --force --batch --runs 8 &
        #time numactl -N 1 -m 1 -- python run.py --dataset ${data} --algorithm scann --local --force --batch --runs 8 &
        #time numactl -N 2 -m 2 -- python run.py --dataset ${data} --algorithm scann --local --force --batch --runs 8 &
        #time numactl -N 3 -m 3 -- python run.py --dataset ${data} --algorithm scann --local --force --batch --runs 8 &
        wait

        python plot.py --dataset ${data} --recompute --batch
      done
  done
}

function ann_b_batch_expand_multiprocess_plot_results() {
  for ((i=1; i<=$1; i++)); do
    #for data in glove-100-angular deep-image-96-angular fashion-mnist-784-euclidean sift-128-euclidean gist-960-euclidean; do
    for data in sift-128-euclidean; do
     python plot.py --dataset ${data} --recompute --batch --x-scale log --y-scale log
    done
  done
}

function main() {

  for i in {1..1}; do
    
    ann_b_batch_expand_multiprocess 1 # 设置batch执行次数
    ann_b_batch_expand_multiprocess_plot_results 1 # 统一打印QPS
    
  done
}

main
