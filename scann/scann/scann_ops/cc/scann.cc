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
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <sstream>
#include <sys/stat.h>

#include "absl/base/internal/sysinfo.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "scann/base/single_machine_base.h"
#include "scann/data_format/dataset.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/partitioning/partitioner.pb.h"
#include "scann/proto/brute_force.pb.h"
#include "scann/proto/centers.pb.h"
#include "scann/proto/scann.pb.h"
#include "scann/scann_ops/scann_assets.pb.h"
#include "scann/tree_x_hybrid/tree_x_params.h"
#include "scann/utils/common.h"
#include "scann/utils/io_npy.h"
#include "scann/utils/io_oss_wrapper.h"
#include "scann/utils/scann_config_utils.h"
#include "scann/utils/single_machine_retraining.h"
#include "scann/utils/threads.h"
#include "scann/utils/types.h"
#include "scann/hw_alg/include/lut16_sse4.h"

namespace research_scann {
namespace {

constexpr const int32_t kSoarEmptyToken = -1;

bool HasSoar(const ScannConfig& config) {
  return config.partitioning().database_spilling().spilling_type() ==
         DatabaseSpillingConfig::TWO_CENTER_ORTHOGONALITY_AMPLIFIED;
}

int GetNumCPUs() { return std::max(absl::base_internal::NumCPUs(), 1); }

unique_ptr<DenseDataset<float>> InitDataset(
    ConstSpan<float> dataset, DatapointIndex n_points,
    DimensionIndex n_dim = kInvalidDimension) {
  if (dataset.empty() && n_dim == kInvalidDimension) return nullptr;

  vector<float> dataset_vec(dataset.data(), dataset.data() + dataset.size());
  auto ds =
      std::make_unique<DenseDataset<float>>(std::move(dataset_vec), n_points);
  if (n_dim != kInvalidDimension) {
    ds->set_dimensionality(n_dim);
  }
  return ds;
}

Status AddTokenizationToOptions(SingleMachineFactoryOptions& opts,
                                ConstSpan<int32_t> tokenization,
                                const int spilling_mult = 1) {
  if (tokenization.empty()) return OkStatus();
  if (opts.serialized_partitioner == nullptr)
    return FailedPreconditionError(
        "Non-empty tokenization but no serialized partitioner is present.");
  opts.datapoints_by_token =
      std::make_shared<vector<std::vector<DatapointIndex>>>(
          opts.serialized_partitioner->n_tokens());
  for (auto [dp_idx, token] : Enumerate(tokenization)) {
    if (token != kSoarEmptyToken) {
      opts.datapoints_by_token->at(token).push_back(dp_idx / spilling_mult);
    }
  }
  return OkStatus();
}

}  // namespace

StatusOr<ScannInterface::ScannArtifacts> ScannInterface::LoadArtifacts(
    const ScannConfig& config, const ScannAssets& orig_assets) {
  ScannAssets assets = orig_assets;
  SingleMachineFactoryOptions opts;

  std::sort(assets.mutable_assets()->pointer_begin(),
            assets.mutable_assets()->pointer_end(),
            [](const ScannAsset* a, const ScannAsset* b) {
              const auto to_int = [](ScannAsset::AssetType a) -> int {
                if (a == ScannAsset::PARTITIONER) return 0;
                if (a == ScannAsset::TOKENIZATION_NPY) return 1;
                return 2 + a;
              };
              return to_int(a->asset_type()) < to_int(b->asset_type());
            });

  unique_ptr<FixedLengthDocidCollection> docids;

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
        
        SCANN_ASSIGN_OR_RETURN(auto pi_vector_and_shape,
                            NumpyToVectorAndShape<float>(pipath));
        pAdaptiveModel->SetProbeInfo(pi_vector_and_shape.first);

        std::string t75path(asset_path);
        std::string t75fn = "train75p.npy";
        t75path.replace(pos, target.length(), t75fn);
        
        SCANN_ASSIGN_OR_RETURN(auto t75_vector_and_shape,
                            NumpyToVectorAndShape<int>(t75path));
        pAdaptiveModel->SetTrain75p(t75_vector_and_shape.first[0]);    
        break;
      }
      case ScannAsset::TOKENIZATION_NPY: {
        SCANN_ASSIGN_OR_RETURN(auto vector_and_shape,
                               NumpyToVectorAndShape<int32_t>(asset_path));
        const int spilling_mult = HasSoar(config) ? 2 : 1;
        SCANN_RETURN_IF_ERROR(AddTokenizationToOptions(
            opts, vector_and_shape.first, spilling_mult));
        if (HasSoar(config)) {
          docids = std::make_unique<FixedLengthDocidCollection>(4);
          docids->Reserve(vector_and_shape.second[0] / 2);

          for (size_t i = 1; i < vector_and_shape.second[0]; i += 2) {
            int32_t token = vector_and_shape.first[i];
            SCANN_RETURN_IF_ERROR(docids->Append(strings::Int32ToKey(token)));
          }
        }
        break;
      }
      case ScannAsset::AH_DATASET_NPY: {
        SCANN_ASSIGN_OR_RETURN(auto vector_and_shape,
                               NumpyToVectorAndShape<uint8_t>(asset_path));
        opts.hashed_dataset = make_shared<DenseDataset<uint8_t>>(
            std::move(vector_and_shape.first), vector_and_shape.second[0]);
        break;
      }
      case ScannAsset::AH_DATASET_SOAR_NPY: {
        DCHECK(HasSoar(config));
        DCHECK_NE(docids, nullptr);
        SCANN_ASSIGN_OR_RETURN(auto vector_and_shape,
                               NumpyToVectorAndShape<uint8_t>(asset_path));
        opts.soar_hashed_dataset = make_shared<DenseDataset<uint8_t>>(
            std::move(vector_and_shape.first), std::move(docids));
        break;
      }
      case ScannAsset::DATASET_NPY: {
        SCANN_ASSIGN_OR_RETURN(auto vector_and_shape,
                               NumpyToVectorAndShape<float>(asset_path));
        dataset = make_shared<DenseDataset<float>>(
            std::move(vector_and_shape.first), vector_and_shape.second[0]);

        if (vector_and_shape.second[0] == 0)
          dataset->set_dimensionality(vector_and_shape.second[1]);
        break;
      }
      case ScannAsset::INT8_DATASET_NPY: {
        SCANN_ASSIGN_OR_RETURN(auto vector_and_shape,
                               NumpyToVectorAndShape<int8_t>(asset_path));
        fp->fixed_point_dataset = make_shared<DenseDataset<int8_t>>(
            std::move(vector_and_shape.first), vector_and_shape.second[0]);

        if (vector_and_shape.second[0] == 0)
          fp->fixed_point_dataset->set_dimensionality(
              vector_and_shape.second[1]);
        break;
      }
      case ScannAsset::INT8_MULTIPLIERS_NPY: {
        SCANN_ASSIGN_OR_RETURN(auto vector_and_shape,
                               NumpyToVectorAndShape<float>(asset_path));
        fp->multiplier_by_dimension =
            make_shared<vector<float>>(std::move(vector_and_shape.first));
        break;
      }
      case ScannAsset::INT8_NORMS_NPY: {
        SCANN_ASSIGN_OR_RETURN(auto vector_and_shape,
                               NumpyToVectorAndShape<float>(asset_path));
        fp->squared_l2_norm_by_datapoint =
            make_shared<vector<float>>(std::move(vector_and_shape.first));
        break;
      }
      // case ScannAsset::BF16_DATASET_NPY: {
      //   SCANN_ASSIGN_OR_RETURN(auto vector_and_shape,
      //                          NumpyToVectorAndShape<int16_t>(asset_path));
      //   opts.bfloat16_dataset = make_shared<DenseDataset<int16_t>>(
      //       std::move(vector_and_shape.first), vector_and_shape.second[0]);
      //   break;
      // }
      default:
        break;
    }
  }
  if (fp->fixed_point_dataset != nullptr) {
    if (fp->squared_l2_norm_by_datapoint == nullptr)
      fp->squared_l2_norm_by_datapoint = make_shared<vector<float>>();
    opts.pre_quantized_fixed_point = fp;
  }
  return std::make_tuple(config, std::move(dataset), std::move(opts));
}

std::string RewriteAssetFilenameIfRelative(const string& artifacts_dir,
                                           const string& asset_path) {
  std::filesystem::path path(asset_path);
  if (path.is_relative()) {
    return (artifacts_dir / path).string();
  } else {
    return asset_path;
  }
}

StatusOr<ScannInterface::ScannArtifacts> ScannInterface::LoadArtifacts(
    const std::string& artifacts_dir, const std::string& scann_assets_pbtxt) {
  ScannConfig config;
  SCANN_RETURN_IF_ERROR(
      ReadProtobufFromFile(artifacts_dir + "/scann_config.pb", &config));
  ScannAssets assets;
  if (scann_assets_pbtxt.empty()) {
    SCANN_ASSIGN_OR_RETURN(auto assets_pbtxt,
                           GetContents(artifacts_dir + "/scann_assets.pbtxt"));
    SCANN_RETURN_IF_ERROR(ParseTextProto(&assets, assets_pbtxt));
  } else {
    SCANN_RETURN_IF_ERROR(ParseTextProto(&assets, scann_assets_pbtxt));
  }
  for (auto i : Seq(assets.assets_size())) {
    auto new_path = RewriteAssetFilenameIfRelative(
        artifacts_dir, assets.assets(i).asset_path());
    assets.mutable_assets(i)->set_asset_path(new_path);
  }
  return LoadArtifacts(config, assets);
}

StatusOr<std::unique_ptr<SingleMachineSearcherBase<float>>>
ScannInterface::CreateSearcher(ScannArtifacts artifacts) {
  auto [config, dataset, opts] = std::move(artifacts);

  if (dataset && config.has_partitioning() &&
      config.partitioning().partitioning_type() ==
          PartitioningConfig::SPHERICAL)
    dataset->set_normalization_tag(research_scann::UNITL2NORM);

  SCANN_ASSIGN_OR_RETURN(auto searcher, SingleMachineFactoryScann<float>(
                                            config, dataset, std::move(opts)));
  searcher->MaybeReleaseDataset();
  return searcher;
}

Status ScannInterface::Initialize(const std::string& config_pbtxt,
                                  const std::string& scann_assets_pbtxt) {
  SCANN_RETURN_IF_ERROR(ParseTextProto(&config_, config_pbtxt));
  ScannAssets assets;
  SCANN_RETURN_IF_ERROR(ParseTextProto(&assets, scann_assets_pbtxt));
  SCANN_ASSIGN_OR_RETURN(auto dataset_and_opts, LoadArtifacts(config_, assets));
  auto [_, dataset, opts] = std::move(dataset_and_opts);
  return Initialize(std::tie(config_, dataset, opts));
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
        std::make_shared<DenseDataset<uint8_t>>(std::move(hashed_db), n_points);
  }
  const int spilling_mult = HasSoar(config_) ? 2 : 1;
  SCANN_RETURN_IF_ERROR(
      AddTokenizationToOptions(opts, datapoint_to_token, spilling_mult));
  if (!int8_dataset.empty()) {
    auto int8_data = std::make_shared<PreQuantizedFixedPoint>();
    vector<int8_t> int8_vec(int8_dataset.data(),
                            int8_dataset.data() + int8_dataset.size());
    int8_data->fixed_point_dataset =
        std::make_shared<DenseDataset<int8_t>>(std::move(int8_vec), n_points);

    int8_data->multiplier_by_dimension = make_shared<vector<float>>(
        int8_multipliers.begin(), int8_multipliers.end());

    int8_data->squared_l2_norm_by_datapoint =
        make_shared<vector<float>>(dp_norms.begin(), dp_norms.end());
    opts.pre_quantized_fixed_point = int8_data;
  }

  DimensionIndex n_dim = kInvalidDimension;
  if (config.input_output().pure_dynamic_config().has_dimensionality())
    n_dim = config.input_output().pure_dynamic_config().dimensionality();
  return Initialize(std::make_tuple(
      config_, InitDataset(dataset, n_points, n_dim), std::move(opts)));
}

Status ScannInterface::Initialize(ConstSpan<float> dataset,
                                  DatapointIndex n_points,
                                  const std::string& config,
                                  int training_threads) {
  std::cout << "313 config = " << config << std::endl;
  SCANN_RETURN_IF_ERROR(ParseTextProto(&config_, config));
  if (training_threads < 0)
    return InvalidArgumentError("training_threads must be non-negative");
  if (training_threads == 0) training_threads = GetNumCPUs();
  SingleMachineFactoryOptions opts;

  opts.parallelization_pool =
      StartThreadPool("scann_threadpool", training_threads - 1);

  DimensionIndex n_dim = kInvalidDimension;
  if (config_.input_output().pure_dynamic_config().has_dimensionality())
    n_dim = config_.input_output().pure_dynamic_config().dimensionality();
  return Initialize(std::make_tuple(
      config_, InitDataset(dataset, n_points, n_dim), std::move(opts)));
}

Status ScannInterface::Initialize(ScannInterface::ScannArtifacts artifacts) {
  auto [config, dataset, opts] = std::move(artifacts);
  config_ = config;
  SCANN_ASSIGN_OR_RETURN(dimensionality_, opts.ComputeConsistentDimensionality(
                                              config_.hash(), dataset.get()));
  int n_points_ = dataset->size();
  SCANN_ASSIGN_OR_RETURN(scann_,
                         CreateSearcher(std::tie(config_, dataset, opts)));
  if (!so_path_.empty()) {
    SetSoPath(so_path_);
  }

  if (scann_->config().has_value()) config_ = scann_->config().value();

  const std::string& distance = config_.distance_measure().distance_measure();
  const absl::flat_hash_set<std::string> negated_distances{
      "DotProductDistance", "BinaryDotProductDistance", "AbsDotProductDistance",
      "LimitedInnerProductDistance"};
  result_multiplier_ =
      negated_distances.find(distance) == negated_distances.end() ? 1 : -1;

  parallel_query_pool_ = StartThreadPool("ScannQueryingPool", GetNumCPUs() - 1);
  bool is_serialized = opts.serialized_partitioner != nullptr;
  if (config_.has_partitioning()) {
    min_batch_size_ = 1;
#ifdef FOR_MILVUS
#else
    if (!is_serialized) {
      std::cout << "----> new pAdaptiveModel and CollectTrainData" << std::endl;
      pAdaptiveModel = adpModelFactory();
      int n_leaves_tmp = config_.partitioning().num_children();
      std::cout << "n_leaves_tmp = " << n_leaves_tmp << std::endl;
      CollectTrainData(&dataset->data()[0], n_points_, dimensionality_, n_leaves_tmp, 50);
    }
#endif
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

StatusOr<ScannConfig> ScannInterface::RetrainAndReindex(const string& config) {
  absl::Mutex mu;
  ScannConfig new_config = config_;
  if (!config.empty())
    SCANN_RETURN_IF_ERROR(ParseTextProto(&new_config, config));

  auto status_or = RetrainAndReindexSearcher(scann_.get(), &mu, new_config,
                                             parallel_query_pool_);
  if (!status_or.ok()) return status_or.status();
  scann_.reset(static_cast<SingleMachineSearcherBase<float>*>(
      std::move(status_or.value().release())));
  if (scann_->config().has_value()) config_ = scann_->config().value();
  scann_->MaybeReleaseDataset();
  return config_;
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
  SCANN_RET_CHECK_EQ(queries.dimensionality(), dimensionality_);
  const size_t numQueries = queries.size();
  const size_t numCPUs = parallel_query_pool_->NumThreads();

  const size_t kBatchSize =
      std::min(std::max(min_batch_size_, DivRoundUp(numQueries, numCPUs)),
               static_cast<size_t>(batch_size));
  return ParallelForWithStatus<1>(
      Seq(DivRoundUp(numQueries, kBatchSize)), parallel_query_pool_.get(),
      [&](size_t i) {
        size_t begin = kBatchSize * i;
        size_t curSize = std::min(numQueries - begin, kBatchSize);
        vector<float> queryCopy(
            queries.data().begin() + begin * dimensionality_,
            queries.data().begin() + (begin + curSize) * dimensionality_);
        DenseDataset<float> curQueryDataset(std::move(queryCopy), curSize);
        return SearchBatched(curQueryDataset, res.subspan(begin, curSize),
                             final_nn, pre_reorder_nn, leaves);
      });
}

StatusOr<ScannAssets> ScannInterface::Serialize(std::string path,
                                                bool relative_path) {
  SCANN_ASSIGN_OR_RETURN(auto opts,
                         scann_->ExtractSingleMachineFactoryOptions());
  ScannAssets assets;
  const auto add_asset = [&assets](const std::string& fpath,
                                   ScannAsset::AssetType type) {
    ScannAsset* asset = assets.add_assets();
    asset->set_asset_type(type);
    asset->set_asset_path(fpath);
  };

  const auto convert_path = [&path, &relative_path](const std::string& fpath) {
    std::string absolute_path = path + "/" + fpath;
    return std::pair(relative_path ? fpath : absolute_path, absolute_path);
  };

  SCANN_RETURN_IF_ERROR(
      WriteProtobufToFile(path + "/scann_config.pb", config_));
  if (opts.ah_codebook != nullptr) {
    auto [rpath, fpath] = convert_path("ah_codebook.pb");
    add_asset(rpath, ScannAsset::AH_CENTERS);
    SCANN_RETURN_IF_ERROR(WriteProtobufToFile(fpath, *opts.ah_codebook));
  }
  if (opts.serialized_partitioner != nullptr) {
    auto [rpath, fpath] = convert_path("serialized_partitioner.pb");
    add_asset(rpath, ScannAsset::PARTITIONER);
    SCANN_RETURN_IF_ERROR(
        WriteProtobufToFile(fpath, *opts.serialized_partitioner));

    auto [rpipath, fpipath] = convert_path("probe_info.npy");
    SCANN_RETURN_IF_ERROR(
        VectorToNumpy(fpipath, pAdaptiveModel->GetProbeInfo()));

    auto [rt75path, ft75path] = convert_path("train75p.npy");
    std::vector<int> t75vec= {pAdaptiveModel->GetTrain75p()};
    SCANN_RETURN_IF_ERROR(
        VectorToNumpy(ft75path, t75vec));

    if (pAdaptiveModel->SaveLibrary(path)){
      std::cout<< "Successfully saved model to " << path << std::endl;
    }   
  }
  if (opts.datapoints_by_token != nullptr) {
    vector<int32_t> datapoint_to_token;
    if (HasSoar(config_)) {
      datapoint_to_token = vector<int32_t>(2 * n_points(), kSoarEmptyToken);
      for (const auto& [token_idx, dps] :
           Enumerate(*opts.datapoints_by_token)) {
        for (auto dp_idx : dps) {
          dp_idx *= 2;
          if (datapoint_to_token[dp_idx] != -1) dp_idx++;
          DCHECK_EQ(datapoint_to_token[dp_idx], -1);
          datapoint_to_token[dp_idx] = token_idx;
        }
      }
    } else {
      datapoint_to_token = vector<int32_t>(n_points());
      for (const auto& [token_idx, dps] : Enumerate(*opts.datapoints_by_token))
        for (auto dp_idx : dps) datapoint_to_token[dp_idx] = token_idx;
    }
    auto [rpath, fpath] = convert_path("datapoint_to_token.npy");
    add_asset(rpath, ScannAsset::TOKENIZATION_NPY);
    SCANN_RETURN_IF_ERROR(VectorToNumpy(fpath, datapoint_to_token));
  }
  if (opts.hashed_dataset != nullptr) {
    auto [rpath, fpath] = convert_path("hashed_dataset.npy");
    add_asset(rpath, ScannAsset::AH_DATASET_NPY);
    SCANN_RETURN_IF_ERROR(DatasetToNumpy(fpath, *(opts.hashed_dataset)));

    if (opts.soar_hashed_dataset != nullptr) {
      DCHECK(HasSoar(config_));
      auto [rpath, fpath] = convert_path("hashed_dataset_soar.npy");
      add_asset(rpath, ScannAsset::AH_DATASET_SOAR_NPY);
      SCANN_RETURN_IF_ERROR(DatasetToNumpy(fpath, *(opts.soar_hashed_dataset)));
    }
  }
  // if (opts.bfloat16_dataset != nullptr) {
  //   auto [rpath, fpath] = convert_path("bfloat16_dataset.npy");
  //   add_asset(rpath, ScannAsset::BF16_DATASET_NPY);
  //   SCANN_RETURN_IF_ERROR(DatasetToNumpy(fpath, *(opts.bfloat16_dataset)));
  // }
  if (opts.pre_quantized_fixed_point != nullptr) {
    auto fixed_point = opts.pre_quantized_fixed_point;
    auto dataset = fixed_point->fixed_point_dataset;
    if (dataset != nullptr) {
      auto [rpath, fpath] = convert_path("int8_dataset.npy");
      add_asset(rpath, ScannAsset::INT8_DATASET_NPY);
      SCANN_RETURN_IF_ERROR(DatasetToNumpy(fpath, *dataset));
    }
    auto multipliers = fixed_point->multiplier_by_dimension;
    if (multipliers != nullptr) {
      auto [rpath, fpath] = convert_path("int8_multipliers.npy");
      add_asset(rpath, ScannAsset::INT8_MULTIPLIERS_NPY);
      SCANN_RETURN_IF_ERROR(VectorToNumpy(fpath, *multipliers));
    }
    auto norms = fixed_point->squared_l2_norm_by_datapoint;
    if (norms != nullptr) {
      auto [rpath, fpath] = convert_path("dp_norms.npy");
      add_asset(rpath, ScannAsset::INT8_NORMS_NPY);
      SCANN_RETURN_IF_ERROR(VectorToNumpy(fpath, *norms));
    }
  }
  SCANN_ASSIGN_OR_RETURN(auto dataset, Float32DatasetIfNeeded());
  if (dataset != nullptr) {
    auto [rpath, fpath] = convert_path("dataset.npy");
    add_asset(rpath, ScannAsset::DATASET_NPY);
    SCANN_RETURN_IF_ERROR(DatasetToNumpy(fpath, *dataset));
  }
  return assets;
}

StatusOr<ScannAssets> ScannInterface::SerializeForAll(std::string path, bool relative_path) {
  StatusOr<ScannAssets> assets_or = Serialize(path, relative_path);
  OpenSourceableFileWriter(path + "/scann_assets.pbtxt").Write(assets_or->DebugString());
  return assets_or;
}

int ScannInterface::SerializeToMemory(uint8_t*& dataPtr, size_t& dataLength) {
  StatusOr<SingleMachineFactoryOptions> status_or_opts = 
      scann_->ExtractSingleMachineFactoryOptions();
  if (!status_or_opts.ok()) {
    return -1;
  }
  
  auto opts = std::move(status_or_opts.value());
  // Memory stream to store the serialized data
  std::ostringstream memory_stream;

  // serialzie config_
  std::string config_serialized;
  if (!config_.SerializeToString(&config_serialized)) {
    return -1;
  }
  uint64_t config_size = config_serialized.size();
  memory_stream.write(reinterpret_cast<const char*>(&config_size), sizeof(config_size));
  memory_stream.write(config_serialized.data(), config_size);

  // helper serialize protobuf
  const auto serialize_protobuf = [&memory_stream](const google::protobuf::Message& proto) -> bool {
    std::string serialized;
    if (!proto.SerializeToString(&serialized)) {
      return false;
    }
    uint64_t size = serialized.size();
    memory_stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
    memory_stream.write(serialized.data(), size);
    return true;
  };

  // helper serialize dataset
  const auto serialize_dataset = [&memory_stream](const auto& dataset) -> bool {
    if (!dataset) return true;
    uint64_t dim = dataset->dimensionality();
    memory_stream.write(reinterpret_cast<const char*>(&dim), sizeof(dim));

    uint64_t n_points = dataset->size();
    memory_stream.write(reinterpret_cast<const char*>(&n_points), sizeof(n_points));

    size_t data_size = dataset->data().size() * sizeof(typename std::decay<decltype(dataset->data()[0])>::type);
    memory_stream.write(reinterpret_cast<const char*>(dataset->data().data()), data_size);
    return true;
  };

#ifdef FOR_MILVUS
#else
  // Serialize ah_codebook if present
  uint8_t has_ah_codebook = (opts.ah_codebook != nullptr) ? 1 : 0;
  memory_stream.write(reinterpret_cast<const char*>(&has_ah_codebook), sizeof(has_ah_codebook));
  if (has_ah_codebook && !serialize_protobuf(*opts.ah_codebook)) {
    return -1;
  }
#endif

  // Serialize serialized_partitioner if present
  uint8_t has_partitioner = (opts.serialized_partitioner != nullptr) ? 1 : 0;
  memory_stream.write(reinterpret_cast<const char*>(&has_partitioner), sizeof(has_partitioner));
  if (has_partitioner) {
      if (!serialize_protobuf(*opts.serialized_partitioner)) {
          return -1;
      }

#ifdef FOR_MILVUS
#else
      // probe_info (vector<float>)
      const auto& probe_info = pAdaptiveModel->GetProbeInfo();
      uint64_t probe_info_size = probe_info.size();
      memory_stream.write(reinterpret_cast<const char*>(&probe_info_size), sizeof(probe_info_size));
      if (probe_info_size > 0) {
          memory_stream.write(reinterpret_cast<const char*>(probe_info.data()), 
                            probe_info_size * sizeof(float));
      }

      // train75p (int)
      int train75p = pAdaptiveModel->GetTrain75p();
      memory_stream.write(reinterpret_cast<const char*>(&train75p), sizeof(train75p));

      // so
      std::string workpath = pAdaptiveModel->GetWorkPath();

      std::string lib_path = workpath;
      lib_path.append("/libadaptivemodel.so");

      uint8_t has_so_file = 0;
      std::vector<char> so_content;

      std::ifstream so_file(lib_path, std::ios::binary);
      if (so_file) {
          has_so_file = 1;
          so_file.seekg(0, std::ios::end);
          size_t so_size = so_file.tellg();
          so_file.seekg(0, std::ios::beg);
          so_content.resize(so_size);
          so_file.read(so_content.data(), so_size);
      } else {
        std::cerr << "Failed to open .so file: " << std::endl;
      }

      memory_stream.write(reinterpret_cast<const char*>(&has_so_file), sizeof(has_so_file));

      if (has_so_file) {
          uint64_t so_size = so_content.size();

          memory_stream.write(reinterpret_cast<const char*>(&so_size), sizeof(so_size));
          memory_stream.write(so_content.data(), so_size);
      }

      Delete(workpath);
#endif
  }

  // Serialize tokenization if present
  uint8_t has_tokenization = (opts.datapoints_by_token != nullptr) ? 1 : 0;
  memory_stream.write(reinterpret_cast<const char*>(&has_tokenization), sizeof(has_tokenization));
  if (has_tokenization) {
    vector<int32_t> datapoint_to_token;
    if (HasSoar(config_)) {
      datapoint_to_token = vector<int32_t>(2 * n_points(), kSoarEmptyToken);
      for (const auto& [token_idx, dps] : Enumerate(*opts.datapoints_by_token)) {
        for (auto dp_idx : dps) {
          dp_idx *= 2;
          if (datapoint_to_token[dp_idx] != -1) dp_idx++;
          DCHECK_EQ(datapoint_to_token[dp_idx], -1);
          datapoint_to_token[dp_idx] = token_idx;
        }
      }
    } else {
      datapoint_to_token = vector<int32_t>(n_points());
      for (const auto& [token_idx, dps] : Enumerate(*opts.datapoints_by_token))
        for (auto dp_idx : dps) datapoint_to_token[dp_idx] = token_idx;
    }
    
    uint64_t token_size = datapoint_to_token.size();
    memory_stream.write(reinterpret_cast<const char*>(&token_size), sizeof(token_size));
    memory_stream.write(reinterpret_cast<const char*>(datapoint_to_token.data()), 
                        token_size * sizeof(int32_t));
  }
  
  // Serialize hashed_dataset if present
  uint8_t has_hashed_dataset = (opts.hashed_dataset != nullptr) ? 1 : 0;
  memory_stream.write(reinterpret_cast<const char*>(&has_hashed_dataset), sizeof(has_hashed_dataset));
  if (has_hashed_dataset && !serialize_dataset(opts.hashed_dataset)) {
    return -1;
  }

  // Serialize soar_hashed_dataset if present
  uint8_t has_soar_hashed_dataset = (opts.soar_hashed_dataset != nullptr) ? 1 : 0;
  memory_stream.write(reinterpret_cast<const char*>(&has_soar_hashed_dataset), sizeof(has_soar_hashed_dataset));
  if (has_soar_hashed_dataset && !serialize_dataset(opts.soar_hashed_dataset)) {
    return -1;
  }
    
  // Serialize float32 dataset if needed
  StatusOr<std::shared_ptr<const DenseDataset<float>>> dataset_status_or = Float32DatasetIfNeeded();
  if (!dataset_status_or.ok()) {
    return -1;
  }
  
  auto dataset = dataset_status_or.value();
  uint8_t has_float32_dataset = (dataset != nullptr) ? 1 : 0;
  memory_stream.write(reinterpret_cast<const char*>(&has_float32_dataset), sizeof(has_float32_dataset));
  if (has_float32_dataset && !serialize_dataset(dataset)) {
    return -1;
  }
  
  std::string data = memory_stream.str();
  dataLength = data.size();
  
  dataPtr = new uint8_t[dataLength];
  if (dataPtr == nullptr) {
    dataLength = 0;
    return -1;
  }
  
  std::memcpy(dataPtr, data.data(), dataLength);
  return 0;
}

int ScannInterface::LoadFromMemory(uint8_t* &dataPtr, size_t &dataLength) {
  if (dataPtr == nullptr || dataLength == 0) {
    std::cout<<"Invalid data pointer or length"<<std::endl;
    return -1;
  }

  std::istringstream memory_stream(std::string(reinterpret_cast<const char*>(dataPtr), dataLength));

  // Helper for reading data from the stream
  const auto read_data = [&memory_stream](auto* data, size_t size) -> bool {
    memory_stream.read(reinterpret_cast<char*>(data), size);
    return memory_stream.good();
  };
  
  // Deserialize config_
  uint64_t config_size;
  if (!read_data(&config_size, sizeof(config_size))) {
    std::cout<<"Failed to read config size"<<std::endl;
    return -1;
  }

  std::string config_serialized(config_size, '\0');
  if (!read_data(&config_serialized[0], config_size)) {
    std::cout<<"Failed to read config data"<<std::endl;
    return -1;
  }
  if (!config_.ParseFromString(config_serialized)) {
    std::cout<<"Failed to parse config"<<std::endl;
    return -1;
  }

  SingleMachineFactoryOptions opts;
  
  // Helper for deserializing protobuf
  const auto deserialize_protobuf = [&memory_stream](auto* proto) -> bool {
    uint64_t size;
    if (!memory_stream.read(reinterpret_cast<char*>(&size), sizeof(size))) {
      return false;
    }
    std::string serialized(size, '\0');
    if (!memory_stream.read(&serialized[0], size)) {
      return false;
    }
    return proto->ParseFromString(serialized);
  };
  
  // Helper for deserializing dataset
  const auto deserialize_dataset = [&memory_stream](auto& dataset_ptr) -> bool {
    using DataType = typename std::decay<decltype(dataset_ptr->data()[0])>::type;
    uint64_t dim;
    if (!memory_stream.read(reinterpret_cast<char*>(&dim), sizeof(dim))) {
      return false;
    }  
    uint64_t n_points;
    if (!memory_stream.read(reinterpret_cast<char*>(&n_points), sizeof(n_points))) {
      return false;
    }
    std::vector<DataType> data(dim * n_points);
    if (!memory_stream.read(reinterpret_cast<char*>(data.data()), 
                         data.size() * sizeof(DataType))) {
      return false;
    }
    dataset_ptr = std::make_shared<DenseDataset<DataType>>(std::move(data), n_points);
    return true;
  };

  // Helper for deserializing vector
  const auto deserialize_vector = [&memory_stream](auto& vec_ptr) -> bool {
    using DataType = typename std::decay<decltype((*vec_ptr)[0])>::type;
    uint64_t size;
    if (!memory_stream.read(reinterpret_cast<char*>(&size), sizeof(size))) {
      return false;
    }
    if (size > 0) {
      vec_ptr = std::make_shared<std::vector<DataType>>(size);
      if (!memory_stream.read(reinterpret_cast<char*>(vec_ptr->data()), 
                           size * sizeof(DataType))) {
        return false;
      }
    } else {
      vec_ptr = std::make_shared<std::vector<DataType>>();
    }    
    return true;
  };

#ifdef FOR_MILVUS
#else
  // Deserialize ah_codebook if present
  uint8_t has_ah_codebook;
  if (!read_data(&has_ah_codebook, sizeof(has_ah_codebook))) {
    std::cout<<"Failed to read ah_codebook flag"<<std::endl;
    return -1;
  }
  if (has_ah_codebook) {
    opts.ah_codebook = std::make_shared<CentersForAllSubspaces>();
    if (!deserialize_protobuf(opts.ah_codebook.get())) {
      std::cout<<"Failed to deserialize ah_codebook"<<std::endl;
      return -1;
    }
  }
#endif

  // Deserialize serialized_partitioner if present
  uint8_t has_partitioner;
  if (!read_data(&has_partitioner, sizeof(has_partitioner))) {
      std::cout<<"Failed to read partitioner flag"<<std::endl;
      return -1;
  }

  if (has_partitioner) {
      opts.serialized_partitioner = std::make_shared<SerializedPartitioner>();
      if (!deserialize_protobuf(opts.serialized_partitioner.get())) {
          std::cout<<"Failed to deserialize partitioner"<<std::endl;
          return -1;
      }
#ifdef FOR_MILVUS
#else
      uint64_t probe_info_size;
      if (!read_data(&probe_info_size, sizeof(probe_info_size))) {
          std::cout<<"Failed to read probe_info size"<<std::endl;
          return -1;
      }
      vector<float> probe_info(probe_info_size);
      if (probe_info_size > 0 && !read_data(probe_info.data(), probe_info_size * sizeof(float))) {
          std::cout<<"Failed to read probe_info data"<<std::endl;
          return -1;
      }

      pAdaptiveModel = adpModelFactory();
      std::string workpath = pAdaptiveModel->GetWorkPath();
      Delete(workpath);

      pAdaptiveModel->SetProbeInfo(probe_info);

      int train75p;
      if (!read_data(&train75p, sizeof(train75p))) {
          std::cout<<"Failed to read train75p"<<std::endl;
          return -1;
      }

      pAdaptiveModel->SetTrain75p(train75p);

      std::string output_path = "/tmp/";
      so_path_ = CreateUuidPath(output_path);

      uint8_t has_so_file;
      memory_stream.read(reinterpret_cast<char*>(&has_so_file), sizeof(has_so_file));

      if (has_so_file) {
          uint64_t so_size;
          memory_stream.read(reinterpret_cast<char*>(&so_size), sizeof(so_size));

          std::vector<char> so_content(so_size);
          memory_stream.read(so_content.data(), so_size);

          std::ofstream out_so(so_path_, std::ios::binary);
          if (!out_so) {
              return -1;
          }
          out_so.write(so_content.data(), so_size);
          chmod(so_path_.c_str(), S_IRUSR | S_IWUSR | S_IXUSR | S_IRGRP | S_IROTH);
      }

      pAdaptiveModel->LoadLibrary(so_path_);
#endif
  }

  unique_ptr<FixedLengthDocidCollection> docids;

  // Deserialize datapoints_by_token if present
  uint8_t has_tokenization;
  if (!read_data(&has_tokenization, sizeof(has_tokenization))) {
    std::cout<<"Failed to read tokenization flag"<<std::endl;
    return -1;
  }

  if (has_tokenization) {
    uint64_t token_size;
    if (!read_data(&token_size, sizeof(token_size))) {
      std::cout<<"Failed to read tokenization size"<<std::endl;
      return -1;
    }
    
    vector<int32_t> datapoint_to_token(token_size);
    if (!read_data(datapoint_to_token.data(), token_size * sizeof(int32_t))) {
      std::cout<<"Failed to read tokenization data"<<std::endl;
      return -1;
    }
    
    const int spilling_mult = HasSoar(config_) ? 2 : 1;
    Status status = AddTokenizationToOptions(opts, datapoint_to_token, spilling_mult);
    size_t shape = datapoint_to_token.size();
 
    if (HasSoar(config_)) {
      docids = std::make_unique<FixedLengthDocidCollection>(4);
      docids->Reserve(shape / 2);
      for (size_t i = 1; i < shape; i += 2) {
        int32_t token = datapoint_to_token[i];
        auto status = docids->Append(strings::Int32ToKey(token));
        if (!status.ok()) {
            return -1;
        }
      }
    }
  }

  // Deserialize hashed_dataset if present
  uint8_t has_hashed_dataset;
  if (!read_data(&has_hashed_dataset, sizeof(has_hashed_dataset))) {
    std::cout<<"Failed to read hashed_dataset flag"<<std::endl;
    return -1;
  }
  
  if (has_hashed_dataset) {
    if (!deserialize_dataset(opts.hashed_dataset)) {
      std::cout<<"Failed to deserialize hashed_dataset"<<std::endl;
      return -1;
    }
  }

  // Deserialize soar_hashed_dataset if present
  uint8_t has_soar_hashed_dataset;
  if (!read_data(&has_soar_hashed_dataset, sizeof(has_soar_hashed_dataset))) {
    std::cout<<"Failed to read soar_hashed_dataset flag"<<std::endl;
    return -1;
  }
  
  if (has_soar_hashed_dataset) {
    using DataType = typename std::decay<decltype(opts.soar_hashed_dataset->data()[0])>::type;
    uint64_t dim;
    if (!memory_stream.read(reinterpret_cast<char*>(&dim), sizeof(dim))) {
      return -1;
    }  
    uint64_t n_points;
    if (!memory_stream.read(reinterpret_cast<char*>(&n_points), sizeof(n_points))) {
      return -1;
    }
    std::vector<DataType> data(dim * n_points);
    if (!memory_stream.read(reinterpret_cast<char*>(data.data()), 
                         data.size() * sizeof(DataType))) {
      return -1;
    }
    opts.soar_hashed_dataset = std::make_shared<DenseDataset<DataType>>(std::move(data),  std::move(docids));
  }

  // Deserialize dataset if present
  uint8_t has_float32_dataset;
  if (!read_data(&has_float32_dataset, sizeof(has_float32_dataset))) {
    std::cout<<"Failed to read float32_dataset flag"<<std::endl;
    return -1;
  }

  shared_ptr<DenseDataset<float>> dataset;
  if (has_float32_dataset) {
    if (!deserialize_dataset(dataset)) {
      std::cout<<"Failed to deserialize float32_dataset"<<std::endl;
      return -1;
    }
  }
 
  Status status = Initialize(std::tie(config_, dataset, opts));
  if (!status.ok()) {
    return -1;
  }
  return 0;
}

StatusOr<SingleMachineFactoryOptions> ScannInterface::ExtractOptions() {
  return scann_->ExtractSingleMachineFactoryOptions();
}
void ScannInterface::findTruth(std::vector<int64_t> &approximateGT, DenseDataset<float> &ptr, int qsize, int n_leaves) {
  NNResultsVector *res = new NNResultsVector[qsize];
  SearchBatchedParallel(ptr, MutableSpan<NNResultsVector>(res, ptr.size()), 10, 1000, n_leaves);
  printf("finish findTruth \n");
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
    // vector<vector<KMeansTreeSearchResult>> qcenters(qsize);
    vector<vector<pair<DatapointIndex, float>>> qcenters(qsize);
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
        vector<pair<DatapointIndex, float>>().swap(i);
    } 
    vector<vector<pair<DatapointIndex, float>>>().swap(qcenters);
    vector<int64_t>().swap(approximateGT);

    std::cout << "--> CollectTrainData trainModel" << std::endl;
    pAdaptiveModel->trainModel(expectedNprobe, fdata, n_leaves);
    pAdaptiveModel->SetMode(IadpModel::AMODE::DISABLE);
}

}  // namespace research_scann
