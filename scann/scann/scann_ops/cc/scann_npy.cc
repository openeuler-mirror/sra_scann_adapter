// Copyright 2023 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "scann/scann_ops/cc/scann_npy.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pybind11/gil.h"
#include "scann/base/single_machine_base.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/scann_ops/cc/scann.h"
#include "scann/utils/common.h"
#include "scann/utils/io_oss_wrapper.h"
#include "scann/utils/types.h"

namespace research_scann {
using MutationOptions = UntypedSingleMachineSearcherBase::MutationOptions;
using PrecomputedMutationArtifacts =
    UntypedSingleMachineSearcherBase::PrecomputedMutationArtifacts;

void RuntimeErrorIfNotOk(const char* prefix, const Status& status) {
  if (!status.ok()) {
    std::string msg = prefix + std::string(status.message());
    throw std::runtime_error(msg);
  }
}

template <typename T>
T ValueOrRuntimeError(StatusOr<T> status_or, const char* prefix) {
  RuntimeErrorIfNotOk(prefix, status_or.status());
  return status_or.value();
}

ScannNumpy::ScannNumpy(const std::string& artifacts_dir,
                       const std::string& scann_assets_pbtxt) {
  auto status_or =
      ScannInterface::LoadArtifacts(artifacts_dir, scann_assets_pbtxt);
  RuntimeErrorIfNotOk("Error loading artifacts: ", status_or.status());
  RuntimeErrorIfNotOk("Error initializing searcher: ",
                      scann_.Initialize(status_or.value()));
}

ScannNumpy::ScannNumpy(const np_row_major_arr<float>& np_dataset,
                       const std::string& config, int training_threads) {
  if (np_dataset.ndim() != 2)
    throw std::invalid_argument("Dataset input must be two-dimensional");
  ConstSpan<float> dataset(np_dataset.data(), np_dataset.size());
  pybind11::gil_scoped_release gil_release;
  RuntimeErrorIfNotOk("Error initializing searcher: ",
                      scann_.Initialize(dataset, np_dataset.shape()[0], config,
                                        training_threads));
}

int ScannNumpy::Rebalance(const string& config) {
  auto statusor = scann_.RetrainAndReindex(config);
  if (!statusor.ok()) {
    RuntimeErrorIfNotOk("Failed to retrain searcher: ", statusor.status());
    return -1;
  }

  return scann_.n_points();
}

size_t ScannNumpy::Size() const { return scann_.n_points(); }

void ScannNumpy::SetNumThreads(int num_threads) {
  scann_.SetNumThreads(num_threads);
}

// string ScannNumpy::Config() {
//   std::string config_str;
//   google::protobuf::TextFormat::PrintToString(*scann_.config(), &config_str);
//   return config_str;
// }

std::pair<pybind11::array_t<DatapointIndex>, pybind11::array_t<float>>
ScannNumpy::Search(const np_row_major_arr<float>& query, int final_nn,
                   int pre_reorder_nn, int leaves) {
  if (query.ndim() != 1)
    throw std::invalid_argument("Query must be one-dimensional");

  DatapointPtr<float> ptr(nullptr, query.data(), query.size(), query.size());
  NNResultsVector res;
  {
    pybind11::gil_scoped_release gil_release;
    auto status = scann_.Search(ptr, &res, final_nn, pre_reorder_nn, leaves);
    RuntimeErrorIfNotOk("Error during search: ", status);
  }

  pybind11::array_t<DatapointIndex> indices(res.size());
  pybind11::array_t<float> distances(res.size());
  auto idx_ptr = reinterpret_cast<DatapointIndex*>(indices.request().ptr);
  auto dis_ptr = reinterpret_cast<float*>(distances.request().ptr);
  scann_.ReshapeNNResult(res, idx_ptr, dis_ptr);
  return {indices, distances};
}

std::pair<pybind11::array_t<DatapointIndex>, pybind11::array_t<float>>
ScannNumpy::SearchBatched(const np_row_major_arr<float>& queries, int final_nn,
                          int pre_reorder_nn, int leaves, bool parallel,
                          int batch_size) {
  if (queries.ndim() != 2)
    throw std::invalid_argument("Queries must be in two-dimensional array");

  vector<float> queries_vec(queries.data(), queries.data() + queries.size());
  auto query_dataset =
      DenseDataset<float>(std::move(queries_vec), queries.shape()[0]);

  std::vector<NNResultsVector> res(query_dataset.size());
  {
    pybind11::gil_scoped_release gil_release;
    Status status;
    if (parallel)
      status = scann_.SearchBatchedParallel(query_dataset, MakeMutableSpan(res),
                                            final_nn, pre_reorder_nn, leaves,
                                            batch_size);
    else
      status = scann_.SearchBatched(query_dataset, MakeMutableSpan(res),
                                    final_nn, pre_reorder_nn, leaves);
    RuntimeErrorIfNotOk("Error during search: ", status);
  }

  for (const auto& nn_res : res)
    final_nn = std::max<int>(final_nn, nn_res.size());
  pybind11::array_t<DatapointIndex> indices(
      {static_cast<long>(query_dataset.size()), static_cast<long>(final_nn)});
  pybind11::array_t<float> distances(
      {static_cast<long>(query_dataset.size()), static_cast<long>(final_nn)});
  auto idx_ptr = reinterpret_cast<DatapointIndex*>(indices.request().ptr);
  auto dis_ptr = reinterpret_cast<float*>(distances.request().ptr);
  scann_.ReshapeBatchedNNResult(MakeConstSpan(res), idx_ptr, dis_ptr, final_nn);
  return {indices, distances};
}

void ScannNumpy::Serialize(std::string path, bool relative_path) {
  StatusOr<ScannAssets> assets_or = scann_.Serialize(path, relative_path);
  RuntimeErrorIfNotOk("Failed to extract SingleMachineFactoryOptions: ",
                      assets_or.status());
  std::string assets_or_text;
  google::protobuf::TextFormat::PrintToString(*assets_or, &assets_or_text);
  RuntimeErrorIfNotOk("Failed to write ScannAssets proto: ",
                      OpenSourceableFileWriter(path + "/scann_assets.pbtxt")
                          .Write(assets_or_text));
}

pybind11::array_t<uint8_t> ScannNumpy::SerializeToNumpy() {
  uint8_t* data_ptr = nullptr;
  size_t data_length = 0;
  
  int result = scann_.SerializeToMemory(data_ptr, data_length);

  if (result == 0) {
    pybind11::capsule free_when_done(data_ptr, [](void* ptr) {
      delete[] reinterpret_cast<uint8_t*>(ptr);
    });

    return pybind11::array_t<uint8_t>(
      {static_cast<pybind11::ssize_t>(data_length)},
      {1},                                    
      data_ptr,                               
      free_when_done  
    );
  }
}

void ScannNumpy::SearchAdditionalParams(float adp_threshold, int adp_refined, int leaves_to_search) {
  scann_.SearchAdditionalParams(adp_threshold, adp_refined, leaves_to_search);
}

int ScannNumpy::GetNum() {
  int total_num = scann_.GetNum();
  if (total_num == 0) {
    std::cout << " Uninitialized " << std::endl;
  }
  return total_num;
}

int ScannNumpy::GetDim() {
  int dim = scann_.GetDim();
  if (dim == 0) {
    std::cout << " Uninitialized " << std::endl;
  } 
  return dim;
}

}  // namespace research_scann
