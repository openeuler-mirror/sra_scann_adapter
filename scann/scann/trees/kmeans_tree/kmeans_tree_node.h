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

#ifndef SCANN_TREES_KMEANS_TREE_KMEANS_TREE_NODE_H_
#define SCANN_TREES_KMEANS_TREE_KMEANS_TREE_NODE_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/distance_measures/one_to_many/one_to_many.h"
#include "scann/distance_measures/one_to_one/l2_distance.h"
#include "scann/oss_wrappers/scann_castops.h"
#include "scann/oss_wrappers/scann_threadpool.h"
#include "scann/proto/partitioning.pb.h"
#include "scann/trees/kmeans_tree/kmeans_tree.pb.h"
#include "scann/trees/kmeans_tree/training_options.h"
#include "scann/utils/common.h"
#include "scann/utils/datapoint_utils.h"
#include "scann/utils/fast_top_neighbors.h"
#include "scann/utils/types.h"
#include "scann/utils/zip_sort.h"
#include "scann/hw_alg/include/L2.h"

namespace research_scann {

class KMeansTreeNode {
 public:
  KMeansTreeNode();

  explicit KMeansTreeNode(int32_t leaf_id) { leaf_id_ = leaf_id; }

  static KMeansTreeNode CreateFlat(DenseDataset<float> centers);

  void Reset();

  bool IsLeaf() const { return children_.empty(); }

  int32_t LeafId() const { return leaf_id_; }

  // template <typename T>
  // Status ApplyAvq(const DenseDataset<T>& dataset,
  //                 ConstSpan<std::vector<DatapointIndex>> datapoints_by_token,
  //                 float avq_eta, ThreadPool* pool_or_null = nullptr);

  const DenseDataset<float>& Centers() const { return float_centers_; }

  const std::vector<KMeansTreeNode>& Children() const { return children_; }

  const std::vector<DatapointIndex>& indices() const { return indices_; }

  void UnionIndices(std::vector<DatapointIndex>* result) const;

  double learned_spilling_threshold() const {
    return learned_spilling_threshold_;
  }

  DatapointPtr<float> cur_node_center() const { return cur_node_center_; }

  template <typename Real>
  const DenseDataset<Real>& GetCentersByTemplateType() const;

 private:
  friend class KMeansTree;

  Status Train(const Dataset& training_data, std::vector<DatapointIndex> subset,
               const DistanceMeasure& training_distance, int32_t k_per_level,
               int32_t current_level, KMeansTreeTrainingOptions* opts);

  template <typename Real, typename DataType>
  Status FindChildrenWithSpilling(
      const DatapointPtr<Real>& query,
      QuerySpillingConfig::SpillingType spilling_type,
      double spilling_threshold, int32_t max_centers,
      const DistanceMeasure& dist,
      std::vector<pair<DatapointIndex, float>>* child_centers) const;

  template <typename Real, typename OutT = double>
  void GetAllDistancesFloatingPoint(const DistanceMeasure& dist,
                                    const DatapointPtr<Real>& query,
                                    std::vector<OutT>* distances) const;
  template <typename OutT = double>
  Status GetAllDistancesInt8(const DistanceMeasure& dist,
                             const DatapointPtr<float>& query,
                             std::vector<OutT>* distances) const;

  void CreateFixedPointCenters();

  void BuildFromProto(const SerializedKMeansTree::Node& proto);

  void CopyToProto(SerializedKMeansTree::Node* proto, bool with_indices) const;

  Status CheckDimensionality(DimensionIndex query_dims) const;

  int32_t NumberLeaves(int32_t m);

  void PopulateCurNodeCenters();

  void UnionIndicesImpl(absl::flat_hash_set<DatapointIndex>* union_hash) const;

  void MaybeInitializeThreadSharding();

  DenseDataset<float> float_centers_;

  std::vector<uint8_t> l2_raw_point_centers_;
 
  double scale_;
  double bias_;

  DenseDataset<int8_t> fixed_point_centers_;

  std::vector<float> inv_int8_multipliers_ = {};

  std::vector<DatapointIndex> indices_ = {};

  std::vector<KMeansTreeNode> children_ = {};

  std::vector<float> center_squared_l2_norms_ = {};

  double learned_spilling_threshold_ = numeric_limits<double>::quiet_NaN();

  int32_t leaf_id_ = -1;

  DatapointPtr<float> cur_node_center_;
};

template <>
inline const DenseDataset<float>&
KMeansTreeNode::GetCentersByTemplateType<float>() const {
  return float_centers_;
}

template <>
inline const DenseDataset<int8_t>&
KMeansTreeNode::GetCentersByTemplateType<int8_t>() const {
  return fixed_point_centers_;
}

namespace kmeans_tree_internal {

template <typename Real, typename OutT>
void GetAllDistancesFloatingPointNoThreadSharding(
    const DistanceMeasure& dist, const DatapointPtr<Real>& query,
    const DenseDataset<Real>& centers, MutableSpan<OutT> distances) {
  if (query.IsDense()) {
    DenseDistanceOneToMany<Real, OutT>(dist, query, centers, distances);
  } else {
    for (size_t i = 0; i < centers.size(); ++i) {
      distances[i] = dist.GetDistance(query, centers[i]);
    }
  }
}

template <typename Real>
StatusOr<Real> ComputeThreshold(
    const Real nearest_center_distance, const Real spilling_threshold,
    QuerySpillingConfig::SpillingType spilling_type) {
  Real max_dist_to_consider;
  if (std::isnan(spilling_threshold)) {
    spilling_type = QuerySpillingConfig::NO_SPILLING;
  }

  switch (spilling_type) {
    case QuerySpillingConfig::NO_SPILLING:
      max_dist_to_consider = nearest_center_distance;
      break;
    case QuerySpillingConfig::MULTIPLICATIVE:
      max_dist_to_consider = nearest_center_distance * spilling_threshold;
      break;
    case QuerySpillingConfig::ADDITIVE:
      max_dist_to_consider = nearest_center_distance + spilling_threshold;
      break;
    case QuerySpillingConfig::ABSOLUTE_DISTANCE:

      max_dist_to_consider =
          std::max(spilling_threshold, nearest_center_distance);
      break;
    case QuerySpillingConfig::FIXED_NUMBER_OF_CENTERS:
      max_dist_to_consider = numeric_limits<Real>::infinity();
      break;
    default:
      return InvalidArgumentError("Unknown spilling type.");
  }
  return max_dist_to_consider;
}

Status PostprocessDistancesForSpilling(
    ConstSpan<float> distances, QuerySpillingConfig::SpillingType spilling_type,
    double spilling_threshold, int32_t max_centers,
    std::vector<pair<DatapointIndex, float>>* child_centers);

}  // namespace kmeans_tree_internal

template <typename Real, typename OutT>
void KMeansTreeNode::GetAllDistancesFloatingPoint(
    const DistanceMeasure& dist, const DatapointPtr<Real>& query,
    std::vector<OutT>* distances) const {
  const auto& centers = float_centers_;

  return kmeans_tree_internal::GetAllDistancesFloatingPointNoThreadSharding(
      dist, query, centers, MakeMutableSpan(*distances));
}

template <typename OutT>
Status KMeansTreeNode::GetAllDistancesInt8(const DistanceMeasure& dist,
                                           const DatapointPtr<float>& query,
                                           std::vector<OutT>* distances) const {
  const auto& centers = fixed_point_centers_;
  const bool is_sq_l2 =
      dist.specially_optimized_distance_tag() == DistanceMeasure::SQUARED_L2;
  if (dist.specially_optimized_distance_tag() != DistanceMeasure::DOT_PRODUCT &&
      !is_sq_l2) {
    return InvalidArgumentError(
        "Fixed-point tokenization in K-Means trees currently works only for "
        "dot-product distance and squared L2 distance.");
  }

  if (is_sq_l2 && scale_ == 1.0 && bias_ == 0.0) {
    int num_centers = centers.size();
    int dims = centers.dimensionality();
 
    uint8_t query_i8[dims];
    for (int i = 0; i < dims; ++i) {
      query_i8[i] = (uint8_t)(query.values()[i] * scale_ + bias_ + 0.5);
    }
    float tt_dis[num_centers];
    hw_alg::L2sqr_batch4_u8InOne(query_i8, l2_raw_point_centers_.data(), num_centers, dims, tt_dis);
    for(int i = 0; i < num_centers; ++i) {
      distances->at(i) = tt_dis[i];
    }
  } else {
    Datapoint<float> inv_adjusted;
    CopyToDatapoint(query, &inv_adjusted);
 
    if (is_sq_l2) {
      for (const auto& [i, inv_mult] : Enumerate(inv_int8_multipliers_))
        inv_adjusted.mutable_values_span()[i] *= inv_mult * 2;
    } else {
      for (const auto& [i, inv_mult] : Enumerate(inv_int8_multipliers_))
        inv_adjusted.mutable_values_span()[i] *= inv_mult;
    }
 
    DenseDotProductDistanceOneToManyInt8Float(inv_adjusted.ToPtr(), centers,
                                              MakeMutableSpan(*distances));
    if (is_sq_l2) {
      DCHECK_EQ(center_squared_l2_norms_.size(), distances->size());
      float query_norm = SquaredL2Norm(query);
      for (const auto [i, center_norm] : Enumerate(center_squared_l2_norms_))
        distances->at(i) += query_norm + center_norm;
    }
  }
  return OkStatus();
}

template <typename Real, typename DataType>
Status KMeansTreeNode::FindChildrenWithSpilling(
    const DatapointPtr<Real>& query,
    QuerySpillingConfig::SpillingType spilling_type, double spilling_threshold,
    int32_t max_centers, const DistanceMeasure& dist,
    std::vector<pair<DatapointIndex, float>>* child_centers) const {
  const auto& centers = this->GetCentersByTemplateType<DataType>();
  DCHECK_GT(centers.size(), 0);
  DCHECK(child_centers);
  SCANN_RET_CHECK(query.IsFinite());

  std::vector<float> distances(centers.size());
  DCHECK(centers.IsDense());

  if constexpr (std::is_floating_point_v<DataType>) {
    this->GetAllDistancesFloatingPoint(dist, query, &distances);
  } else {
    static_assert(std::is_same_v<DataType, int8_t>);
    SCANN_RETURN_IF_ERROR(this->GetAllDistancesInt8(dist, query, &distances));
  }

  return kmeans_tree_internal::PostprocessDistancesForSpilling(
      MakeMutableSpan(distances), spilling_type, spilling_threshold,
      max_centers, child_centers);
}

}  // namespace research_scann

#endif
