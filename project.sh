#!/bin/bash

workdir=$(cd $(dirname $0); pwd)
echo "workdir=" $workdir
export LD_LIBRARY_PATH=$workdir/ksl/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$workdir/hw_alg/lib:$LD_LIBRARY_PATH

prepare(){
    cd $workdir/scann
    python configure.py
}

build_py(){
    cd $workdir/scann
    CC=gcc bazel build -c opt --linkopt="-fuse-ld=gold" --cxxopt="-std=c++17" --copt=-fsized-deallocation --copt=-w --copt=-g --cxxopt=-g --copt=-O3 --cxxopt=-O3 --copt=-fprefetch-loop-arrays --cxxopt=-fprefetch-loop-arrays --copt=-march=armv8.2-a+lse+sve+dotprod --cxxopt=-march=armv8.2-a+lse+sve+dotprod --copt=-mtune=tsv110 --cxxopt=-mtune=tsv110 :build_pip_pkg
    bazel-bin/build_pip_pkg > /dev/null
}

build_eval_bazel_sve(){
    cd /usr/local/ksl/lib
    rm -rf libavx2ki.so
    ln -s libavx2sve.so libavx2ki.so
    cd $workdir/scann
    CC=gcc CXX=g++ bazel build -c opt --linkopt="-fuse-ld=gold" --cxxopt="-std=c++17" --copt=-fsized-deallocation --copt=-w --copt=-g --cxxopt=-g --copt=-O3 --cxxopt=-O3 --copt=-fprefetch-loop-arrays --cxxopt=-fprefetch-loop-arrays --copt=-march=armv8.2-a+lse+sve+dotprod --cxxopt=-march=armv8.2-a+lse+sve+dotprod --copt=-mtune=tsv110 --cxxopt=-mtune=tsv110 :eval
    cp ./bazel-bin/eval ./
}

build_eval_bazel_neon(){
    cd /usr/local/ksl/lib
    rm -rf libavx2ki.so
    ln -s libavx2neon.so libavx2ki.so
    cd $workdir/scann
    CC=gcc CXX=g++ bazel build -c opt --linkopt="-fuse-ld=gold" --cxxopt="-std=c++17" --copt=-fsized-deallocation --copt=-w --copt=-g --cxxopt=-g --copt=-O3 --cxxopt=-O3 --copt=-fprefetch-loop-arrays --cxxopt=-fprefetch-loop-arrays --copt=-march=armv8.2-a+lse+dotprod --cxxopt=-march=armv8.2-a+lse+dotprod --copt=-mtune=tsv110 --cxxopt=-mtune=tsv110 :eval
    cp ./bazel-bin/eval ./
}

build_scann_cc_sve(){
    cd /usr/local/ksl/lib
    rm -rf libavx2ki.so
    ln -s libavx2sve.so libavx2ki.so
    cd $workdir/scann
    CC=gcc bazel build -c opt --linkopt="-fuse-ld=gold" --linkopt="-static-libgcc" --linkopt="-static-libstdc++" --cxxopt="-std=c++17" --copt=-DSCANN_SVE --cxxopt=-DSCANN_SVE --copt=-fsized-deallocation --copt=-w --copt=-g --cxxopt=-g --copt=-O3 --cxxopt=-O3 --copt=-fprefetch-loop-arrays --cxxopt=-fprefetch-loop-arrays --copt=-march=armv8.2-a+lse+sve+dotprod  --cxxopt=-march=armv8.2-a+lse+sve+dotprod --copt=-mtune=tsv110 --cxxopt=-mtune=tsv110 //scann/scann_ops/cc:libscann_cc.so
    chmod +w bazel-out/aarch64-opt/bin/scann/scann_ops/cc/libscann_cc.so-2.params
    sed -i '/-lstdc++/d' bazel-out/aarch64-opt/bin/scann/scann_ops/cc/libscann_cc.so-2.params
    /opt/openEuler/gcc-toolset-12/root/usr/bin/gcc @bazel-out/aarch64-opt/bin/scann/scann_ops/cc/libscann_cc.so-2.params
    cp ./bazel-bin/scann/scann_ops/cc/libscann_cc.so ./
}

build_scann_cc_neon(){
    cd /usr/local/ksl/lib
    rm -rf libavx2ki.so
    ln -s libavx2neon.so libavx2ki.so
    cd $workdir/scann
    CC=gcc bazel build -c opt --linkopt="-fuse-ld=gold" --linkopt="-static-libgcc" --linkopt="-static-libstdc++" --cxxopt="-std=c++17" --copt=-fsized-deallocation --copt=-w --copt=-g --cxxopt=-g --copt=-O3 --cxxopt=-O3 --copt=-fprefetch-loop-arrays --cxxopt=-fprefetch-loop-arrays --copt=-march=armv8.2-a+lse+dotprod  --cxxopt=-march=armv8.2-a+lse+dotprod --copt=-mtune=tsv110 --cxxopt=-mtune=tsv110 //scann/scann_ops/cc:libscann_cc.so
    chmod +w bazel-out/aarch64-opt/bin/scann/scann_ops/cc/libscann_cc.so-2.params
    sed -i '/-lstdc++/d' bazel-out/aarch64-opt/bin/scann/scann_ops/cc/libscann_cc.so-2.params
    /opt/openEuler/gcc-toolset-12/root/usr/bin/gcc @bazel-out/aarch64-opt/bin/scann/scann_ops/cc/libscann_cc.so-2.params
    cp ./bazel-bin/scann/scann_ops/cc/libscann_cc.so ./
}

build_scann_cc_sve_milvus(){
    cd /usr/local/ksl/lib
    rm -rf libavx2ki.so
    ln -s libavx2sve.so libavx2ki.so
    cd $workdir/scann
    CC=gcc bazel build -c opt --linkopt="-fuse-ld=gold" --linkopt="-static-libgcc" --linkopt="-static-libstdc++" --cxxopt="-std=c++17" --copt=-DSCANN_SVE --cxxopt=-DSCANN_SVE --copt=-DFOR_MILVUS --cxxopt=-DFOR_MILVUS --copt=-fsized-deallocation --copt=-w --copt=-g --cxxopt=-g --copt=-O3 --cxxopt=-O3 --copt=-fprefetch-loop-arrays --cxxopt=-fprefetch-loop-arrays --copt=-march=armv8.2-a+lse+sve+dotprod  --cxxopt=-march=armv8.2-a+lse+sve+dotprod --copt=-mtune=tsv110 --cxxopt=-mtune=tsv110 //scann/scann_ops/cc:libscann_cc.so
    chmod +w bazel-out/aarch64-opt/bin/scann/scann_ops/cc/libscann_cc.so-2.params
    sed -i '/-lstdc++/d' bazel-out/aarch64-opt/bin/scann/scann_ops/cc/libscann_cc.so-2.params
    /opt/openEuler/gcc-toolset-12/root/usr/bin/gcc @bazel-out/aarch64-opt/bin/scann/scann_ops/cc/libscann_cc.so-2.params
    cp ./bazel-bin/scann/scann_ops/cc/libscann_cc.so ./
}

build_scann_cc_neon_milvus(){
    cd /usr/local/ksl/lib
    rm -rf libavx2ki.so
    ln -s libavx2neon.so libavx2ki.so
    cd $workdir/scann
    CC=gcc bazel build -c opt --linkopt="-fuse-ld=gold" --linkopt="-static-libgcc" --linkopt="-static-libstdc++" --cxxopt="-std=c++17" --copt=-DFOR_MILVUS --cxxopt=-DFOR_MILVUS --copt=-fsized-deallocation --copt=-w --copt=-g --cxxopt=-g --copt=-O3 --cxxopt=-O3 --copt=-fprefetch-loop-arrays --cxxopt=-fprefetch-loop-arrays --copt=-march=armv8.2-a+lse+dotprod  --cxxopt=-march=armv8.2-a+lse+dotprod --copt=-mtune=tsv110 --cxxopt=-mtune=tsv110 //scann/scann_ops/cc:libscann_cc.so
    chmod +w bazel-out/aarch64-opt/bin/scann/scann_ops/cc/libscann_cc.so-2.params
    sed -i '/-lstdc++/d' bazel-out/aarch64-opt/bin/scann/scann_ops/cc/libscann_cc.so-2.params
    /opt/openEuler/gcc-toolset-12/root/usr/bin/gcc @bazel-out/aarch64-opt/bin/scann/scann_ops/cc/libscann_cc.so-2.params
    cp ./bazel-bin/scann/scann_ops/cc/libscann_cc.so ./
}

build_eval_cmake_sve(){
    cd /usr/local/ksl/lib
    rm -rf libavx2ki.so
    ln -s libavx2sve.so libavx2ki.so
    cd $workdir/scann
    rm -rf build
    mkdir -p build && cd build
    cmake .. && make
    cp eval_cmake $workdir/scann
}

build_eval_cmake_neon(){
    cd /usr/local/ksl/lib
    rm -rf libavx2ki.so
    ln -s libavx2neon.so libavx2ki.so
    cd $workdir/scann
    rm -rf build
    mkdir -p build && cd build
    cmake .. && make
    cp eval_cmake $workdir/scann
}

 
OPTIONS=abcdefghij
LONGOPTIONS=prepare,build_whl,build_eval_bazel_sve,build_eval_bazel_neon,build_scann_cc_sve,build_scann_cc_neon,build_scann_cc_sve_milvus,build_scann_cc_neon_milvus,build_eval_cmake_sve,build_eval_cmake_neon

PARSED=$(getopt -o $OPTIONS --long $LONGOPTIONS -n "$0" -- "$@")

if [ $? -ne 0 ]; then
    exit 1
fi

eval set -- "$PARSED"

while true; do
    case "$1" in
        -a|--prepare)
        echo "-----------------------------------------------------------"
        echo "--> prepare python dependency: " $(TZ=UTC-8 date +%Y-%m-%d" "%H:%M:%S)
        echo "-----------------------------------------------------------"
        prepare
        shift
        ;;
        -b|--build_whl)
        echo "-----------------------------------------------------------"
        echo "--> build whl: " $(TZ=UTC-8 date +%Y-%m-%d" "%H:%M:%S)
        echo "-----------------------------------------------------------"
        build_py
        shift
        ;;
        -c|--build_eval_bazel_sve)
        echo "-----------------------------------------------------------"
        echo "--> build eval with bazel(sve): " $(TZ=UTC-8 date +%Y-%m-%d" "%H:%M:%S)
        echo "-----------------------------------------------------------"
        build_eval_bazel_sve
        shift
        ;;
        -d|--build_eval_bazel_neon)
        echo "-----------------------------------------------------------"
        echo "--> build eval with bazel(neon): " $(TZ=UTC-8 date +%Y-%m-%d" "%H:%M:%S)
        echo "-----------------------------------------------------------"
        build_eval_bazel_neon
        shift
        ;;
        -e|--build_scann_cc_sve)
        echo "-----------------------------------------------------------"
        echo "--> build libscann_cc.so(sve)注意替换hw_alg下的lib与include: " $(TZ=UTC-8 date +%Y-%m-%d" "%H:%M:%S)
        echo "-----------------------------------------------------------"
        build_scann_cc_sve
        shift
        ;;
        -f|--build_scann_cc_neon)
        echo "-----------------------------------------------------------"
        echo "--> build libscann_cc.so(neon)注意替换hw_alg下的lib与include: " $(TZ=UTC-8 date +%Y-%m-%d" "%H:%M:%S)
        echo "-----------------------------------------------------------"
        build_scann_cc_neon
        shift
        ;;
        -g|--build_scann_cc_sve_milvus)
        echo "-----------------------------------------------------------"
        echo "--> build libscann_cc.so(sve milvus)注意替换hw_alg下的lib与include: " $(TZ=UTC-8 date +%Y-%m-%d" "%H:%M:%S)
        echo "-----------------------------------------------------------"
        build_scann_cc_sve_milvus
        shift
        ;;
        -h|--build_scann_cc_neon_milvus)
        echo "-----------------------------------------------------------"
        echo "--> build libscann_cc.so(neon milvus)注意替换hw_alg下的lib与include: " $(TZ=UTC-8 date +%Y-%m-%d" "%H:%M:%S)
        echo "-----------------------------------------------------------"
        build_scann_cc_neon_milvus
        shift
        ;;
        -i|--build_eval_cmake_sve)
        echo "-----------------------------------------------------------"
        echo "--> build eval with cmake(sve): " $(TZ=UTC-8 date +%Y-%m-%d" "%H:%M:%S)
        echo "-----------------------------------------------------------"
        build_eval_cmake_sve
        shift
        ;;
        -j|--build_eval_cmake_neon)
        echo "-----------------------------------------------------------"
        echo "--> build eval with cmake(neon): " $(TZ=UTC-8 date +%Y-%m-%d" "%H:%M:%S)
        echo "-----------------------------------------------------------"
        build_eval_cmake_neon
        shift
        ;;
        --)
        shift
        break
        ;;
        *)
        echo "unsupported param."
        exit 1
        ;;
    esac
done
