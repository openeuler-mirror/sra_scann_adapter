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

#include "scann/hashes/asymmetric_hashing2/querying.h"

#include <cstdint>
#include <utility>

#include "scann/utils/common.h"
#include "scann/utils/intrinsics/flags.h"
#include "scann/hw_alg/include/kscann.h"


using std::shared_ptr;

namespace research_scann {
namespace asymmetric_hashing2 {

PackedDataset _CreatePackedDataset(
    const DenseDataset<uint8_t>& hashed_database) {
  PackedDataset result;
  result.bit_packed_data =
      asymmetric_hashing_internal::CreatePackedDataset(hashed_database);
  result.num_datapoints = hashed_database.size();

  // only works for even number of dimensions (num_blocks).
  // For odd value of num_blocks, one should adjust packed dataset generation and Sse4LUT16BottomLoop
  assert(hashed_database.empty() || hashed_database[0].nonzero_entries() % 2 == 0);
  result.num_blocks =
      (!hashed_database.empty()) ? ((hashed_database[0].nonzero_entries() + 1) & (~1)) : 0;
  return result;
}

DenseDataset<uint8_t> UnpackDataset(const PackedDataset &packed)
{
  return DenseDataset<uint8_t>(
      hw_alg::UnpackDatasetImpl(packed.bit_packed_data, packed.num_blocks, packed.num_datapoints),
      packed.num_datapoints);
}

template <typename T>
AsymmetricQueryer<T>::AsymmetricQueryer(
    shared_ptr<const ChunkingProjection<T>> projector,
    shared_ptr<const DistanceMeasure> lookup_distance,
    shared_ptr<const Model<T>> model)
    : projector_(std::move(projector)),
      lookup_distance_(std::move(lookup_distance)),
      model_(std::move(model)) {}

template <typename T>
StatusOr<LookupTable> AsymmetricQueryer<T>::CreateLookupTable(
    const DatapointPtr<T>& query,
    AsymmetricHasherConfig::LookupType lookup_type,
    AsymmetricHasherConfig::FixedPointLUTConversionOptions
        float_int_conversion_options) const {
  switch (lookup_type) {
    case AsymmetricHasherConfig::FLOAT:
      return CreateLookupTable<float>(query, float_int_conversion_options);
    case AsymmetricHasherConfig::INT8:
    case AsymmetricHasherConfig::INT8_LUT16:
      return CreateLookupTable<int8_t>(query, float_int_conversion_options);
    case AsymmetricHasherConfig::INT16:
      return CreateLookupTable<int16_t>(query, float_int_conversion_options);
    default:
      return InvalidArgumentError("Unrecognized lookup type.");
  }
}

SCANN_INSTANTIATE_TYPED_CLASS(, AsymmetricQueryer);

}  // namespace asymmetric_hashing2
}  // namespace research_scann
