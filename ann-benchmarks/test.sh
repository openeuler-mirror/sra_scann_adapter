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

function prepare_scann() {
  cd ${project_path}/scann
  git_info
  git diff
  pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
  pip config set global.trusted_host mirrors.aliyun.com
  python configure.py
  pip install tensorflow
  bazel clean;
}

function prepare_ann_benchmark() {
  cd ${project_path}/ann-benchmarks
  git_info
  git diff
  pip install -r requirements.txt
  yum install -y python3-numpy python3-scipy python3-pip git
  pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple/
  pip3 config set global.trusted_host mirrors.aliyun.com
  pip3 install -U pip
  pip3 install -r requirements.txt
}

function compile_scann() {
  echo "compile $1 scann..."

  ${GCC} -v

  cd ${project_path}/scann
  git_info

  # 配置并行时获取的CPU数
  sed -n '41p' ./scann/scann_ops/cc/scann.cc
  sed -i '41s/.*/int GetNumCPUs() { return std::max(absl::base_internal::NumCPUs()\/'$1', 1); }/' ./scann/scann_ops/cc/scann.cc
  sed -n '41p' ./scann/scann_ops/cc/scann.cc

  extra_opt=''
  for opt in "${@:3}"; do
    if [ "$opt" == "inst-lib-inline" ]; then
      extra_opt+=' --copt=-finline-mm --cxxopt=-finline-mm'
    elif [ "$opt" == "920-tune" ]; then
      extra_opt+=' --copt=-march=armv8.2-a+lse --cxxopt=-march=armv8.2-a+lse --copt=-mtune=tsv110 --cxxopt=-mtune=tsv110 --copt=-fprefetch-loop-arrays --cxxopt=-fprefetch-loop-arrays'
    elif [ "$opt" == "920b-tune" ]; then
      extra_opt+=' --copt=-march=armv8.2-a+lse+sve --cxxopt=-march=armv8.2-a+lse+sve --copt=-mtune=tsv110 --cxxopt=-mtune=tsv110 --copt=-fprefetch-loop-arrays --cxxopt=-fprefetch-loop-arrays'
    elif [ "$opt" == "calculate" ]; then
      extra_opt+=' --copt=--param --copt=remove-redundant-and=1 --cxxopt=--param --cxxopt=remove-redundant-and=1 --copt=-fdump-tree-aprefetch-details --cxxopt=-fdump-tree-aprefetch-details
      --copt=--param --copt=optimized-calculation=1 --cxxopt=--param --cxxopt=optimized-calculation=1 --copt=-fdump-tree-einline-details --cxxopt=-fdump-tree-einline-details'
    elif [ "$opt" == "prefetch-t3" ]; then
      extra_opt+=' --copt=--param --copt=customized-issue=1 --cxxopt=--param --cxxopt=customized-issue=1 --copt=--param --copt=customized-ahead-factor=6 --cxxopt=--param --cxxopt=customized-ahead-factor=6 --copt=--param --copt=customized-temporal=3 --cxxopt=--param --cxxopt=customized-temporal=3 --copt=-fdump-tree-aprefetch-details --cxxopt=-fdump-tree-aprefetch-details'
    elif [ "$opt" == "prefetch-t0" ]; then
      extra_opt+=' --copt=--param --copt=customized-issue=1 --cxxopt=--param --cxxopt=customized-issue=1 --copt=--param --copt=customized-ahead-factor=6 --cxxopt=--param --cxxopt=customized-ahead-factor=6 --copt=--param --copt=customized-temporal=0 --cxxopt=--param --cxxopt=customized-temporal=0 --copt=-fdump-tree-aprefetch-details --cxxopt=-fdump-tree-aprefetch-details'
    elif [ "$opt" == "prefetch-dataset" ]; then
      extra_opt+=' --copt=--param --copt=customized-dataset-issue=1 --cxxopt=--param --cxxopt=customized-dataset-issue=1 --copt=-fdump-tree-einline-details --cxxopt=-fdump-tree-einline-details'
    elif [ "$opt" == "inline-tune" ]; then
      extra_opt+='
      --copt=--param --copt=max-early-inliner-iterations=5 --cxxopt=--param --cxxopt=max-early-inliner-iterations=5
      --copt=--param --copt=max-inline-functions-called-once-loop-depth=30 --cxxopt=--param --cxxopt=max-inline-functions-called-once-loop-depth=30
      --copt=--param --copt=max-inline-functions-called-once-insns=200000  --cxxopt=--param --cxxopt=max-inline-functions-called-once-insns=200000
      --copt=--param --copt=max-inline-insns-auto=75  --cxxopt=--param --cxxopt=max-inline-insns-auto=75
      --copt=--param --copt=max-inline-insns-recursive=2250  --cxxopt=--param --cxxopt=max-inline-insns-recursive=2250
      --copt=--param --copt=max-inline-insns-recursive-auto=2250  --cxxopt=--param --cxxopt=max-inline-insns-recursive-auto=2250
      --copt=--param --copt=max-inline-insns-single=35000  --cxxopt=--param --cxxopt=max-inline-insns-single=35000
      --copt=--param --copt=max-inline-recursive-depth=40  --cxxopt=--param --cxxopt=max-inline-recursive-depth=40
      --copt=--param --copt=max-inline-recursive-depth-auto=40  --cxxopt=--param --cxxopt=max-inline-recursive-depth-auto=40
      --copt=-fdump-tree-einline-details --cxxopt=-fdump-tree-einline-details
      --copt=-fdump-ipa-inline-details --cxxopt=-fdump-ipa-inline-details
      '
    else
      echo "Error: Invalid option input parameter."
      return 1
    fi
  done

  platform=$2
  if [ "$platform" == "arm_base" ]; then
    CC=${GCC} bazel build -c opt --cxxopt="-std=c++17" --copt=-fsized-deallocation --copt=-w --copt=-g --cxxopt=-g --copt=-O3 --cxxopt=-O3 ${extra_opt} :build_pip_pkg && ./bazel-bin/build_pip_pkg && pip3 uninstall scann -y && pip3 install scann-1.2.10-cp39-cp39-linux_aarch64.whl
  elif [ "$platform" == "x86_64" ]; then
    CC=${GCC} bazel build -c opt --copt=-mavx --copt=-mfma --cxxopt="-std=c++17" --copt=-fsized-deallocation --copt=-w --copt=-g --cxxopt=-g --copt=-O3 --cxxopt=-O3 :build_pip_pkg && ./bazel-bin/build_pip_pkg && ./bazel-bin/build_pip_pkg && pip3 uninstall scann -y && pip3 install scann-1.2.10-cp39-cp39-linux_x86_64.whl
    #CC=clang bazel build -c opt --features=thin_lto --copt=-mavx --copt=-mfma --cxxopt="-std=c++17" --copt=-fsized-deallocation --copt=-w --copt=-g --cxxopt=-g --copt=-O3 --cxxopt=-O3 :build_pip_pkg && ./bazel-bin/build_pip_pkg && ./bazel-bin/build_pip_pkg && pip3 uninstall scann -y && pip3 install scann-1.2.10-cp39-cp39-linux_x86_64.whl
  else
    echo "Error: Invalid platform input parameter."
    return 1
  fi

  echo ""
}

# 多进程并行 parallelism
function ann_b_parallelism() {
  cd ${project_path}/ann-benchmarks
  git_info

  for core in $@; do
    echo ""
    echo "============ test parallelism ${core} ================"
    echo ""

    cd ${project_path}/ann-benchmarks/ann_benchmarks/algorithms/scann
    python3 scann_config_gen.py ${core}
    cp config-${core}.yml config.yml

    rm -rf ${project_path}/ann-benchmarks/results/glove-100-angular/10/scann
    cd ${project_path}/ann-benchmarks
    time python run.py --dataset glove-100-angular --algorithm scann --local --force --parallelism ${core}

    python plot.py --dataset glove-100-angular --recompute
  done
}

function expand_query() {
  # 扩大query倍数
  sed -n '122p' ./ann_benchmarks/runner.py
  sed -i '122s/.*/            results = batch_query(X_test, '$1')  # expanded n times/' ./ann_benchmarks/runner.py
  sed -n '122p' ./ann_benchmarks/runner.py
}

# 多线程并行 batch
function ann_b_batch() {
  cd ${project_path}/ann-benchmarks
    git_info

  expand_query 1

  for ((i=1; i<=$1; i++)); do
    echo ""
    echo "============ test batch times ${i} ================"
    echo ""

    cd ${project_path}/ann-benchmarks/ann_benchmarks/algorithms/scann
    cp config-org.yml config.yml

    rm -rf ${project_path}/ann-benchmarks/results/glove-100-angular/10/scann-batch
    cd ${project_path}/ann-benchmarks
    time python run.py --dataset glove-100-angular --algorithm scann --local --force --batch

    python plot.py --dataset glove-100-angular --recompute --batch
  done
}

function ann_b_batch_expand() {
  cd ${project_path}/ann-benchmarks
    git_info

  expand_query 400

  for ((i=1; i<=$1; i++)); do
    for data in glove-100-angular deep-image-96-angular fashion-mnist-784-euclidean sift-128-euclidean gist-960-euclidean; do
      echo ""
      echo "============ test batch times ${i} ================"
      echo ""

      cd ${project_path}/ann-benchmarks/ann_benchmarks/algorithms/scann
      cp config-${data}.yml config.yml

      rm -rf ${project_path}/ann-benchmarks/results/${data}/10/scann-batch
      cd ${project_path}/ann-benchmarks
      time python run.py --dataset ${data} --algorithm scann --local --force --batch

      python plot.py --dataset ${data} --recompute --batch
    done
  done
}

function ann_b_batch_expand_multiprocess() {
  cd ${project_path}/ann-benchmarks
    git_info

  expand_query 100

  for ((i=1; i<=$1; i++)); do
    for data in glove-100-angular deep-image-96-angular fashion-mnist-784-euclidean sift-128-euclidean gist-960-euclidean; do
      echo ""
      echo "============ test batch times ${i} ================"
      echo ""

      cd ${project_path}/ann-benchmarks/ann_benchmarks/algorithms/scann
      cp config-${data}.yml config.yml

      rm -rf ${project_path}/ann-benchmarks/results/${data}/10/scann-batch
      cd ${project_path}/ann-benchmarks

      # 目前仅支持ARM配置
      time numactl -N 0 -m 0 -- python run.py --dataset ${data} --algorithm scann --local --force --batch --runs 8&
      time numactl -N 1 -m 1 -- python run.py --dataset ${data} --algorithm scann --local --force --batch --runs 8&
      time numactl -N 2 -m 2 -- python run.py --dataset ${data} --algorithm scann --local --force --batch --runs 8&
      time numactl -N 3 -m 3 -- python run.py --dataset ${data} --algorithm scann --local --force --batch --runs 8&
      wait

      python plot.py --dataset ${data} --recompute --batch
    done
  done
}

function ann_b_batch_expand_multiprocess_plot_results() {
  for ((i=1; i<=$1; i++)); do
    for data in glove-100-angular deep-image-96-angular fashion-mnist-784-euclidean sift-128-euclidean gist-960-euclidean; do
     python plot.py --dataset ${data} --recompute --batch --x-scale log --y-scale log
    done
  done
}

function main() {
  #lscpu
  #cat /etc/os-release
  #cat /proc/version
  #echo "Contents of the current script:"
  #cat $0
  #echo "End of Script"

  for i in {1..1}; do
    #prepare_scann
    #prepare_ann_benchmarks

    # 根据使用的编译器指定路径
    #echo "============ switch gcc-12 ================"
    #GCC=${project_path}/gcc/install-12/bin/gcc
    #export LD_LIBRARY_PATH=${project_path}/gcc/install-12/lib64:$LD_LIBRARY_PATH

    # 根据硬件平台切换对应分支
    #echo "============ checkout arm_migrate ================" 
    #cd ${project_path}/scann
    #git checkout arm_migrate

    # 根据硬件平台切换对应的编译选项
    #echo "============ test arm_base 920-tune calculate prefetch ================"
    #compile_scann 1 arm_base inst-lib-inline 920b-tune calculate prefetch-t3 # 根据平台修改
    #ann_b_parallelism 159 # 设置parallelism核心数
    #ann_b_batch 1 # 设置batch执行次数

    #echo "============ test arm_base 920-tune calculate prefetch ================"
    #compile_scann 1 arm_base inst-lib-inline 920b-tune calculate prefetch-t3 # 根据平台修改
    #ann_b_batch_expand 1 # 设置batch执行次数

    #echo "============ test arm_base 920-tune calculate prefetch ================"
    #compile_scann 4 arm_base inst-lib-inline 920b-tune calculate prefetch-t3 # 根据平台修改
    
    ann_b_batch_expand_multiprocess 1 # 设置batch执行次数
    ann_b_batch_expand_multiprocess_plot_results 1 # 统一打印QPS
    
  done
}

main
