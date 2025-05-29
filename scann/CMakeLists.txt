cmake_minimum_required(VERSION 3.12)
project(scann_eval LANGUAGES C CXX)

#------------------------------- 基础设置 -------------------------------#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_VERBOSE_MAKEFILE ON)

add_compile_options(
  -O3
  -fprefetch-loop-arrays
  -march=armv8.2-a+lse+sve+dotprod
  -mtune=tsv110
  -fsized-deallocation
  -w
  -g
  -fopenmp
  -static-libgcc
  -static-libstdc++
)

#------------------------------- 依赖项查找 -------------------------------#
find_package(OpenMP REQUIRED)
#find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(HDF5 REQUIRED)
#find_package(Protobuf REQUIRED)
find_package(absl REQUIRED COMPONENTS base status statusor strings synchronization flat_hash_set)

set(Protobuf_ROOT "/usr/local/protobuf-3.21.9")
find_package(Protobuf REQUIRED 
  HINTS "${Protobuf_ROOT}"
  COMPONENTS libprotobuf
  NO_DEFAULT_PATH
)
set(PROTOBUF_INCLUDE_DIR "${Protobuf_ROOT}/include")

set(EIGEN3_INCLUDE_DIR "/usr/local/eigen-3.3.7/include/eigen3")

set(KSL_INCLUDE_DIR "/usr/local/ksl/include/")

set(SRA_SOURCE_DIR "/usr/local/sra_recall/lib/kscann")

#------------------------------- 包含路径 -------------------------------#
include_directories(
  ${CMAKE_SOURCE_DIR}
  # 直接包含 Bazel 生成的头文件路径
  ${CMAKE_SOURCE_DIR}/bazel-bin
  ${HDF5_INCLUDE_DIRS}
  ${PROTOBUF_INCLUDE_DIR}
  ${absl_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${KSL_INCLUDE_DIR}
)

#------------------------------- 可执行文件 -------------------------------#
add_executable(eval_cmake eval.cpp cmdline.h)

#------------------------------- 链接配置 -------------------------------#
target_link_options(eval_cmake PRIVATE -static-libgcc -static-libstdc++)
target_link_libraries(eval_cmake PRIVATE
  ${CMAKE_SOURCE_DIR}/libscann_cc.so
  ${SRA_SOURCE_DIR}/libkscann.so
  OpenMP::OpenMP_CXX
  #HDF5::HDF5
  ${HDF5_C_LIBRARIES}
  absl::statusor
  absl::status
  absl::base
  absl::strings
  absl::flat_hash_set
  absl::synchronization
  protobuf::libprotobuf
  pthread
  uuid
)

set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)