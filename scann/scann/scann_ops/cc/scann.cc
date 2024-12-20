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

#include "scann/scann_ops/cc/scann.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/internal/sysinfo.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_set.h"
#include "scann/partitioning/partitioner.pb.h"
#include "scann/proto/brute_force.pb.h"
#include "scann/proto/centers.pb.h"
#include "scann/tree_x_hybrid/tree_x_params.h"
#include "scann/utils/io_npy.h"
#include "scann/utils/io_oss_wrapper.h"
#include "scann/utils/scann_config_utils.h"
#include "scann/utils/threads.h"
#include "scann/hw_alg/include/lut16_sse4.h"

namespace research_scann {
namespace {

int GetNumCPUs() { return std::max(absl::base_internal::NumCPUs(), 1); }

template <typename T>
Status ParseTextProto(T* proto, const string& proto_str) {
  ::google::protobuf::TextFormat::ParseFromString(proto_str, proto);
  return OkStatus();
}

unique_ptr<DenseDataset<float>> InitDataset(ConstSpan<float> dataset,
                                            DatapointIndex n_points) {
  if (dataset.empty()) return nullptr;

  vector<float> dataset_vec(dataset.data(), dataset.data() + dataset.size());
  return std::make_unique<DenseDataset<float>>(dataset_vec, n_points);
}

Status AddTokenizationToOptions(SingleMachineFactoryOptions& opts,
                                ConstSpan<int32_t> tokenization) {
  if (tokenization.empty()) return OkStatus();
  if (opts.serialized_partitioner == nullptr)
    return FailedPreconditionError(
        "Non-empty tokenization but no serialized partitioner is present.");
  opts.datapoints_by_token =
      std::make_shared<vector<std::vector<DatapointIndex>>>(
          opts.serialized_partitioner->n_tokens());
  for (auto [dp_idx, token] : Enumerate(tokenization))
    opts.datapoints_by_token->at(token).push_back(dp_idx);
  return OkStatus();
}

}  // namespace

Status ScannInterface::Initialize(const std::string& config_pbtxt,
                                  const std::string& scann_assets_pbtxt) {
  SCANN_RETURN_IF_ERROR(ParseTextProto(&config_, config_pbtxt));

  SingleMachineFactoryOptions opts;
  ScannAssets assets;
  //LOG(INFO) << "before ParseTextProto" << std::endl;
  SCANN_RETURN_IF_ERROR(ParseTextProto(&assets,scann_assets_pbtxt));
  //LOG(INFO) << "ParseTextProto success" << std::endl;
  shared_ptr<DenseDataset<float>> dataset;
  auto fp = make_shared<PreQuantizedFixedPoint>();
  for (const ScannAsset& asset : assets.assets()) {
    const string_view asset_path = asset.asset_path();
    switch (asset.asset_type()) {
      case ScannAsset::AH_CENTERS:
        opts.ah_codebook = std::make_shared<CentersForAllSubspaces>();
        SCANN_RETURN_IF_ERROR(
            ReadProtobufFromFile(asset_path, opts.ah_codebook.get()));
        break;
      case ScannAsset::PARTITIONER: {
        opts.serialized_partitioner = std::make_shared<SerializedPartitioner>();
        SCANN_RETURN_IF_ERROR(ReadProtobufFromFile(
            asset_path, opts.serialized_partitioner.get()));
            
        std::string target = "serialized_partitioner.pb";
        std::string str(asset_path);
        std::string replacement = "libadaptivemodel.so";
        size_t pos = str.find(target);
        if (pos != std::string::npos) {
          str.replace(pos, target.length(), replacement);
        }
        pAdaptiveModel = adpModelFactory();
        pAdaptiveModel->LoadLibrary(str);

        std::string pipath(asset_path);
        std::string pifn = "probe_info.npy";
        pipath.replace(pos, target.length(), pifn);
        
        TF_ASSIGN_OR_RETURN(auto pi_vector_and_shape,
                            NumpyToVectorAndShape<float>(pipath));
        pAdaptiveModel->SetProbeInfo(pi_vector_and_shape.first);

        std::string t75path(asset_path);
        std::string t75fn = "train75p.npy";
        t75path.replace(pos, target.length(), t75fn);
        
        TF_ASSIGN_OR_RETURN(auto t75_vector_and_shape,
                            NumpyToVectorAndShape<int>(t75path));
        pAdaptiveModel->SetTrain75p(t75_vector_and_shape.first[0]);
        break;
      }
      case ScannAsset::TOKENIZATION_NPY: {
        TF_ASSIGN_OR_RETURN(auto vector_and_shape,
                            NumpyToVectorAndShape<int32_t>(asset_path));
        SCANN_RETURN_IF_ERROR(
            AddTokenizationToOptions(opts, vector_and_shape.first));
        break;
      }
      case ScannAsset::AH_DATASET_NPY: {
        TF_ASSIGN_OR_RETURN(auto vector_and_shape,
                            NumpyToVectorAndShape<uint8_t>(asset_path));
        opts.hashed_dataset = make_shared<DenseDataset<uint8_t>>(
            std::move(vector_and_shape.first), vector_and_shape.second[0]);
        break;
      }
      case ScannAsset::DATASET_NPY: {
        TF_ASSIGN_OR_RETURN(auto vector_and_shape,
                            NumpyToVectorAndShape<float>(asset_path));
        dataset = make_shared<DenseDataset<float>>(
            std::move(vector_and_shape.first), vector_and_shape.second[0]);
        break;
      }
      case ScannAsset::INT8_DATASET_NPY: {
        TF_ASSIGN_OR_RETURN(auto vector_and_shape,
                            NumpyToVectorAndShape<int8_t>(asset_path));
        fp->fixed_point_dataset = make_shared<DenseDataset<int8_t>>(
            std::move(vector_and_shape.first), vector_and_shape.second[0]);
        break;
      }
      case ScannAsset::INT8_MULTIPLIERS_NPY: {
        TF_ASSIGN_OR_RETURN(auto vector_and_shape,
                            NumpyToVectorAndShape<float>(asset_path));
        fp->multiplier_by_dimension =
            make_shared<vector<float>>(std::move(vector_and_shape.first));
        break;
      }
      case ScannAsset::INT8_NORMS_NPY: {
        TF_ASSIGN_OR_RETURN(auto vector_and_shape,
                            NumpyToVectorAndShape<float>(asset_path));
        fp->squared_l2_norm_by_datapoint =
            make_shared<vector<float>>(std::move(vector_and_shape.first));
        break;
      }
      default:
        break;
    }
  }
  if (fp->fixed_point_dataset != nullptr) {
    if (fp->squared_l2_norm_by_datapoint == nullptr)
      fp->squared_l2_norm_by_datapoint = make_shared<vector<float>>();
    opts.pre_quantized_fixed_point = fp;
  }
  return Initialize(dataset, opts);
}

Status ScannInterface::Initialize(
    ScannConfig config, SingleMachineFactoryOptions opts,
    ConstSpan<float> dataset, ConstSpan<int32_t> datapoint_to_token,
    ConstSpan<uint8_t> hashed_dataset, ConstSpan<int8_t> int8_dataset,
    ConstSpan<float> int8_multipliers, ConstSpan<float> dp_norms,
    DatapointIndex n_points) {
  config_ = config;
  if (opts.ah_codebook != nullptr) {
    vector<uint8_t> hashed_db(hashed_dataset.data(),
                              hashed_dataset.data() + hashed_dataset.size());
    opts.hashed_dataset =
        std::make_shared<DenseDataset<uint8_t>>(hashed_db, n_points);
  }
  SCANN_RETURN_IF_ERROR(AddTokenizationToOptions(opts, datapoint_to_token));
  if (!int8_dataset.empty()) {
    auto int8_data = std::make_shared<PreQuantizedFixedPoint>();
    vector<int8_t> int8_vec(int8_dataset.data(),
                            int8_dataset.data() + int8_dataset.size());
    int8_data->fixed_point_dataset =
        std::make_shared<DenseDataset<int8_t>>(int8_vec, n_points);

    int8_data->multiplier_by_dimension = make_shared<vector<float>>(
        int8_multipliers.begin(), int8_multipliers.end());

    int8_data->squared_l2_norm_by_datapoint =
        make_shared<vector<float>>(dp_norms.begin(), dp_norms.end());
    opts.pre_quantized_fixed_point = int8_data;
  }
  return Initialize(InitDataset(dataset, n_points), opts);
}

Status ScannInterface::Initialize(ConstSpan<float> dataset,
                                  DatapointIndex n_points,
                                  const std::string& config,
                                  int training_threads) {
  SCANN_RETURN_IF_ERROR(ParseTextProto(&config_, config));
  if (training_threads < 0)
    return InvalidArgumentError("training_threads must be non-negative");
  if (training_threads == 0) training_threads = GetNumCPUs();
  SingleMachineFactoryOptions opts;

  opts.parallelization_pool =
      StartThreadPool("scann_threadpool", training_threads - 1);
  return Initialize(InitDataset(dataset, n_points), opts);
}

Status ScannInterface::Initialize(shared_ptr<DenseDataset<float>> dataset,
                                  SingleMachineFactoryOptions opts) {
  TF_ASSIGN_OR_RETURN(dimensionality_, opts.ComputeConsistentDimensionality(
                                           config_.hash(), dataset.get()));
  TF_ASSIGN_OR_RETURN(n_points_, opts.ComputeConsistentSize(dataset.get()));

  if (dataset && config_.has_partitioning() &&
      config_.partitioning().partitioning_type() ==
          PartitioningConfig::SPHERICAL)
    dataset->set_normalization_tag(research_scann::UNITL2NORM);

  bool is_serialized = opts.serialized_partitioner != nullptr;
  TF_ASSIGN_OR_RETURN(scann_, SingleMachineFactoryScann<float>(
                                  config_, dataset, std::move(opts)));

  const std::string& distance = config_.distance_measure().distance_measure();
  const absl::flat_hash_set<std::string> negated_distances{
      "DotProductDistance", "BinaryDotProductDistance", "AbsDotProductDistance",
      "LimitedInnerProductDistance"};
  result_multiplier_ =
      negated_distances.find(distance) == negated_distances.end() ? 1 : -1;

  parallel_query_pool_ = StartThreadPool("ScannQueryingPool", GetNumCPUs() - 1);

  if (config_.has_partitioning()) {
    min_batch_size_ = 1;
    if (!is_serialized) {
      std::cout << "----> new pAdaptiveModel and CollectTrainData" << std::endl;
      pAdaptiveModel = adpModelFactory();
      int n_leaves_tmp = config_.partitioning().num_children();
      CollectTrainData(&dataset->data()[0], n_points_, dimensionality_, n_leaves_tmp, 50);
    }
  } else {
    if (config_.has_hash())
      min_batch_size_ = 16;
    else
      min_batch_size_ = 256;
  }
  
  return OkStatus();
}

SearchParameters ScannInterface::GetSearchParameters(int final_nn,
                                                     int pre_reorder_nn,
                                                     int leaves) const {
  SearchParameters params;
  bool has_reordering = config_.has_exact_reordering();
  int post_reorder_nn = -1;
  if (has_reordering) {
    post_reorder_nn = final_nn;
  } else {
    pre_reorder_nn = final_nn;
  }
  params.set_pre_reordering_num_neighbors(pre_reorder_nn);
  params.set_post_reordering_num_neighbors(post_reorder_nn);
  if (leaves > 0) {
    auto tree_params = std::make_shared<TreeXOptionalParameters>();
    tree_params->set_num_partitions_to_search_override(leaves);
    params.set_searcher_specific_optional_parameters(tree_params);
  }
  return params;
}

vector<SearchParameters> ScannInterface::GetSearchParametersBatched(
    int batch_size, int final_nn, int pre_reorder_nn, int leaves,
    bool set_unspecified) const {
  vector<SearchParameters> params(batch_size);
  bool has_reordering = config_.has_exact_reordering();
  int post_reorder_nn = -1;
  if (has_reordering) {
    post_reorder_nn = final_nn;
  } else {
    pre_reorder_nn = final_nn;
  }
  std::shared_ptr<research_scann::TreeXOptionalParameters> tree_params;
  if (leaves > 0) {
    tree_params = std::make_shared<TreeXOptionalParameters>();
    tree_params->set_num_partitions_to_search_override(leaves);
  }

  for (auto& p : params) {
    p.set_pre_reordering_num_neighbors(pre_reorder_nn);
    p.set_post_reordering_num_neighbors(post_reorder_nn);
    if (tree_params) p.set_searcher_specific_optional_parameters(tree_params);
    if (set_unspecified) scann_->SetUnspecifiedParametersToDefaults(&p);
  }
  return params;
}

Status ScannInterface::Search(const DatapointPtr<float> query,
                              NNResultsVector* res, int final_nn,
                              int pre_reorder_nn, int leaves) const {
  if (query.dimensionality() != dimensionality_)
    return InvalidArgumentError("Query doesn't match dataset dimsensionality");
  SearchParameters params =
      GetSearchParameters(final_nn, pre_reorder_nn, leaves);
  scann_->SetUnspecifiedParametersToDefaults(&params);
  return scann_->FindNeighbors(query, params, res);
}

Status ScannInterface::SearchBatched(const DenseDataset<float>& queries,
                                     MutableSpan<NNResultsVector> res,
                                     int final_nn, int pre_reorder_nn,
                                     int leaves) const {
  if (queries.dimensionality() != dimensionality_)
    return InvalidArgumentError("Query doesn't match dataset dimsensionality");
  if (!std::isinf(scann_->default_pre_reordering_epsilon()) ||
      !std::isinf(scann_->default_post_reordering_epsilon()))
    return InvalidArgumentError("Batch querying isn't supported with epsilon");
  auto params = GetSearchParametersBatched(queries.size(), final_nn,
                                           pre_reorder_nn, leaves, true);
  return scann_->FindNeighborsBatched(queries, params, MakeMutableSpan(res));
}

Status ScannInterface::SearchBatchedParallel(const DenseDataset<float>& queries,
                                             MutableSpan<NNResultsVector> res,
                                             int final_nn, int pre_reorder_nn,
                                             int leaves, int batch_size) const {
  const size_t numQueries = queries.size();
  const size_t numCPUs = parallel_query_pool_->NumThreads() <= 0 ?
                                 1 : parallel_query_pool_->NumThreads();
  const size_t kBatchSize = std::min(
      std::max(min_batch_size_, DivRoundUp(numQueries, numCPUs)), static_cast<size_t>(batch_size));

  return ParallelForWithStatus<1>(
      Seq(DivRoundUp(numQueries, kBatchSize)), parallel_query_pool_.get(),
      [&](size_t i) {
        size_t begin = kBatchSize * i;
        size_t curSize = std::min(numQueries - begin, kBatchSize);
        vector<float> queryCopy(
            queries.data().begin() + begin * dimensionality_,
            queries.data().begin() + (begin + curSize) * dimensionality_);
        DenseDataset<float> curQueryDataset(queryCopy, curSize);
        return SearchBatched(curQueryDataset, res.subspan(begin, curSize),
                             final_nn, pre_reorder_nn, leaves);
      });
}

StatusOr<ScannAssets> ScannInterface::Serialize(std::string path) {
  TF_ASSIGN_OR_RETURN(auto opts, scann_->ExtractSingleMachineFactoryOptions());
  ScannAssets assets;
  const auto add_asset = [&assets](const std::string& fpath,
                                   ScannAsset::AssetType type) {
    ScannAsset* asset = assets.add_assets();
    asset->set_asset_type(type);
    asset->set_asset_path(fpath);
  };

  SCANN_RETURN_IF_ERROR(
      WriteProtobufToFile(path + "/scann_config.pb", &config_));
  if (opts.ah_codebook != nullptr) {
    std::string fpath = path + "/ah_codebook.pb";
    add_asset(fpath, ScannAsset::AH_CENTERS);
    SCANN_RETURN_IF_ERROR(WriteProtobufToFile(fpath, opts.ah_codebook.get()));
  }
  if (opts.serialized_partitioner != nullptr) {
    std::string fpath = path + "/serialized_partitioner.pb";
    add_asset(fpath, ScannAsset::PARTITIONER);
    SCANN_RETURN_IF_ERROR(
        WriteProtobufToFile(fpath, opts.serialized_partitioner.get()));

    std::string pipath = path + "/probe_info.npy";
    SCANN_RETURN_IF_ERROR(
        VectorToNumpy(pipath, pAdaptiveModel->GetProbeInfo()));

    std::string t75path = path + "/train75p.npy";
    std::vector<int> t75vec= {pAdaptiveModel->GetTrain75p()};
    SCANN_RETURN_IF_ERROR(
        VectorToNumpy(t75path, t75vec));

    if (pAdaptiveModel->SaveLibrary(path)){
      std::cout<< "Successfully saved model to " << path << std::endl;
    }
  }
  if (opts.datapoints_by_token != nullptr) {
    vector<int32_t> datapoint_to_token(n_points_);
    for (const auto& [token_idx, dps] : Enumerate(*opts.datapoints_by_token))
      for (auto dp_idx : dps) datapoint_to_token[dp_idx] = token_idx;
    std::string fpath = path + "/datapoint_to_token.npy";
    add_asset(fpath, ScannAsset::TOKENIZATION_NPY);
    SCANN_RETURN_IF_ERROR(VectorToNumpy(fpath, datapoint_to_token));
  }
  if (opts.hashed_dataset != nullptr) {
    std::string fpath = path + "/hashed_dataset.npy";
    add_asset(fpath, ScannAsset::AH_DATASET_NPY);
    SCANN_RETURN_IF_ERROR(DatasetToNumpy(fpath, *(opts.hashed_dataset)));
  }
  if (opts.pre_quantized_fixed_point != nullptr) {
    auto fixed_point = opts.pre_quantized_fixed_point;
    auto dataset = fixed_point->fixed_point_dataset;
    if (dataset != nullptr) {
      std::string fpath = path + "/int8_dataset.npy";
      add_asset(fpath, ScannAsset::INT8_DATASET_NPY);
      SCANN_RETURN_IF_ERROR(DatasetToNumpy(fpath, *dataset));
    }
    auto multipliers = fixed_point->multiplier_by_dimension;
    if (multipliers != nullptr) {
      std::string fpath = path + "/int8_multipliers.npy";
      add_asset(fpath, ScannAsset::INT8_MULTIPLIERS_NPY);
      SCANN_RETURN_IF_ERROR(VectorToNumpy(fpath, *multipliers));
    }
    auto norms = fixed_point->squared_l2_norm_by_datapoint;
    if (norms != nullptr) {
      std::string fpath = path + "/dp_norms.npy";
      add_asset(fpath, ScannAsset::INT8_NORMS_NPY);
      SCANN_RETURN_IF_ERROR(VectorToNumpy(fpath, *norms));
    }
  }
  TF_ASSIGN_OR_RETURN(auto dataset, Float32DatasetIfNeeded());
  if (dataset != nullptr) {
    std::string fpath = path + "/dataset.npy";
    add_asset(fpath, ScannAsset::DATASET_NPY);
    SCANN_RETURN_IF_ERROR(DatasetToNumpy(fpath, *dataset));
  }
  std::string fpath = path + "/assets.pb";
  SCANN_RETURN_IF_ERROR(WriteProtobufToFile(fpath, &assets));
  return assets;
}

StatusOr<SingleMachineFactoryOptions> ScannInterface::ExtractOptions() {
  return scann_->ExtractSingleMachineFactoryOptions();
}

void ScannInterface::findTruth(std::vector<int64_t> &approximateGT, DenseDataset<float> &ptr, int qsize, int n_leaves) {
  NNResultsVector *res = new NNResultsVector[qsize];
  SearchBatchedParallel(ptr, MutableSpan<NNResultsVector>(res, ptr.size()), 10, 1000, n_leaves);
  for (int i = 0; i < qsize; ++i) {
      for (auto &pir: res[i]) {
          approximateGT.emplace_back(pir.first);
      }
  }
}
// scann has been Initialized
void ScannInterface::CollectTrainData(const float *pdata, int nb, int dim, int n_leaves, int nprobe) {

    auto CalFullRecall = [](int64_t *a, int64_t *b, int size) {
      std::unordered_set<int64_t> s;
      for (int i = 0; i < size; i++) {
          s.insert(a[i]);
          s.insert(b[i]);
      }
      return (float(size * 2 - s.size()) / float(size));
    };

    // normal search paramenters
    int topK = 10;
    const int actualProbe = nprobe;
    int reorderMax = topK * 100;

    // dataset characteristic parameters
    const int nProbeMax = n_leaves;
    int qsize = nb < 50000 ? nb : 50000;
    pAdaptiveModel->SetMode(IadpModel::AMODE::COLLECT);

    // build train query
    DenseDataset<float> ptr(std::vector<float>(pdata, pdata + dim * qsize), qsize);
    std::cout << "--> CollectTrainData" << std::endl;
    // find ground truth using n_leaves
    std::vector<int64_t> approximateGT;
    findTruth(approximateGT, ptr, qsize, n_leaves);

    // binary Search expectedNprobe
    std::vector<int64_t> expectedNprobe(qsize);
    const float maxRecall = 0.9999;
    vector<vector<KMeansTreeSearchResult>> qcenters(qsize);
    ParallelFor<1>(
      Seq(qsize), parallel_query_pool_.get(), [&](size_t i) {
        int npMin = 1;
        int npMax = nProbeMax;
        int curNprobe = (npMin + npMax) / 2;

        std::vector<int64_t> temp_ids(topK);
        int64_t * curGt = approximateGT.data() + i * topK;

        DenseDataset<float> ptr2(std::vector<float>(pdata + dim * i, pdata + dim * i + dim * 1), 1);
        NNResultsVector *res2 = new NNResultsVector[1];
        // SearchBatched(ptr2, MutableSpan<NNResultsVector>(res2, ptr2.size()), topK, reorderMax, nProbeMax);
        while (npMin <= npMax) {
            curNprobe = (npMin + npMax) / 2;
            SearchBatched(ptr2, MutableSpan<NNResultsVector>(res2, ptr2.size()), topK, reorderMax, curNprobe);
            int idx = 0;
            for (auto &pir: res2[0]) {
                temp_ids[idx] = pir.first;
                idx += 1;
            }
            float rate = CalFullRecall(&temp_ids[0], curGt, topK);
            if (rate < maxRecall) {
                npMin = curNprobe + 1;
            } else if (rate > maxRecall) {
                npMax = curNprobe - 1;
            } else {
                expectedNprobe[i] = curNprobe;
            }
        }
        if (expectedNprobe[i] == 0) {
            expectedNprobe[i] = npMax + 1;
        }

        DatapointPtr<float> query(nullptr, pdata + dim * i, dim, dim);
        scann_->SearchCenters(query, actualProbe, qcenters[i]);
    });

    vector<vector<Entry>> fdata;
    fdata.resize(qsize);
    for (int i=0; i<qsize; i++) {
        ADP_COLLECT_DATA(fdata[i], qcenters[i], actualProbe);
    }

    for(auto& i :qcenters){
        vector<KMeansTreeSearchResult>().swap(i);
    } 
    vector<vector<KMeansTreeSearchResult>>().swap(qcenters);
    vector<int64_t>().swap(approximateGT);

    std::cout << "--> CollectTrainData trainModel" << std::endl;
    pAdaptiveModel->trainModel(expectedNprobe, fdata, n_leaves);
    pAdaptiveModel->SetMode(IadpModel::AMODE::DISABLE);
}

}  // namespace research_scann
