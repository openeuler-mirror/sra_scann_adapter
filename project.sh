#!/bin/bash

# example: sh project.sh -pbut

workdir=$(cd $(dirname $0); pwd)
echo "workdir=" $workdir
export LD_LIBRARY_PATH=$workdir/ksl/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$workdir/hw_alg/lib:$LD_LIBRARY_PATH
prepare(){
    cd $workdir/scann

    # python configure.py will fail under in docker containers
    # BEGIN FIX
    _CURRENT_INDEX_URL=$(pip3 config get global.index-url)
    if [[ "$_CURRENT_INDEX_URL" == *"mirrors.tools.huawei.com/pypi/simple"* ]] && ! pip3 show h5py &> /dev/null; then
        #under docker 
        pip3 install h5py==3.10.0
    fi
    # ENDFIX

    python configure.py
}

build_py(){
    cd $workdir/scann
    #902B(SVE)
    CC=gcc bazel build -c opt --cxxopt="-std=c++17" --copt=-fsized-deallocation --copt=-w --copt=-fstack-protector-strong --copt=-g --cxxopt=-g --copt=-O3 --cxxopt=-O3 --copt=-fprefetch-loop-arrays --cxxopt=-fprefetch-loop-arrays --copt=-march=armv8.2-a+lse+sve --cxxopt=-march=armv8.2-a+lse+sve --copt=-mtune=tsv110 --cxxopt=-mtune=tsv110 :build_pip_pkg
    #CC=gcc bazel build -c opt --cxxopt="-std=c++17" --copt=-fsized-deallocation --copt=-w --copt=-g --cxxopt=-g --copt=-O3 --cxxopt=-O3 --copt=-Om --cxxopt=-Om :build_pip_pkg
    #920()
    #CC=gcc bazel build -c opt --cxxopt="-std=c++17" --copt=-fsized-deallocation --copt=-w --copt=-g --cxxopt=-g --copt=-O3 --cxxopt=-O3 --copt=-fprefetch-loop-arrays --copt=-march=armv8.2-a+lse --cxxopt=-march=armv8.2-a+lse --copt=-mtune=tsv110 --cxxopt=-mtune=tsv110 :build_pip_pkg
    bazel-bin/build_pip_pkg > /dev/null
}

update_py(){
    cd $workdir/scann
    pip3 uninstall scann -y -q
    pip3 install scann-*.whl -q
}

test_py(){
    cd $workdir/ann-benchmarks
    pip3 install -r requirements.txt
    bash test.sh
}

while getopts "apbut" opt
do
    case $opt in
        a)
        shift;;
        p)
        echo "-----------------------------------------------------------"
        echo "--> prepare python dependency: " $(TZ=UTC-8 date +%Y-%m-%d" "%H:%M:%S)
        echo "-----------------------------------------------------------"
        prepare
        ;;
        b)
        echo "-----------------------------------------------------------"
        echo "--> build: " $(TZ=UTC-8 date +%Y-%m-%d" "%H:%M:%S)
        echo "-----------------------------------------------------------"
        build_py
        ;;
        u)
        echo "-----------------------------------------------------------"
        echo "--> update python package: " $(TZ=UTC-8 date +%Y-%m-%d" "%H:%M:%S)
        echo "-----------------------------------------------------------"
        update_py
        ;;
        t)
        echo "-----------------------------------------------------------"
        echo "--> run: " $(TZ=UTC-8 date +%Y-%m-%d" "%H:%M:%S)
        echo "-----------------------------------------------------------"
        test_py
        ;;
        ?)
        echo "unsupported param."
        exit 1
        ;;
    esac
done
