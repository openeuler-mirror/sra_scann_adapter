#include <cstdint>

#include "scann/hashes/internal/lut16_sse4.h"
#include "scann/oss_wrappers/scann_bits.h"
#include "scann/utils/common.h"

#ifdef __aarch64__

#include "absl/base/prefetch.h"
#include "scann/utils/bits.h"
#include "scann/utils/intrinsics/sse4.h"
#include "scann/hw_alg/include/lut16_sse4.h" 

namespace research_scann {
namespace asymmetric_hashing_internal {
namespace {

template <size_t size, typename T>
SCANN_INLINE array<T, size> ToLocalArray(ConstSpan<T> span) {
  DCHECK_EQ(span.size(), size);
  array<T, size> result;
  std::copy(span.begin(), span.begin() + size, result.begin());
  return result;
}

template <size_t kNumQueries, PrefetchStrategy kPrefetch>
SCANN_SSE4_INLINE Sse4<int16_t, kNumQueries, 4> Sse4LUT16BottomLoop(
    const uint8_t* data_start, array<const uint8_t*, kNumQueries> lookup_starts,
    DimensionIndex num_blocks) {
  uint16x8x4_t acc[kNumQueries];
  int16x8_t total_bias = vdupq_n_s16(num_blocks * 128);
  // std::cout << "start Sse4LUT16BottomLoopExtractNew" << std::endl;
  hw_alg::Sse4LUT16BottomLoopExtractNew<kNumQueries>(acc,data_start,lookup_starts,num_blocks);
  
  Sse4<int16_t, kNumQueries, 4> int16_accums = sse4::Zeros();
  for (size_t j : Seq(kNumQueries)) {
    int16_accums[j][0] = {.vect_s16 = vsubq_s16(vreinterpretq_s16_u16(acc[j].val[0]), total_bias)};
    int16_accums[j][1] = {.vect_s16 = vsubq_s16(vreinterpretq_s16_u16(acc[j].val[1]), total_bias)};
    int16_accums[j][2] = {.vect_s16 = vsubq_s16(vreinterpretq_s16_u16(acc[j].val[2]), total_bias)};
    int16_accums[j][3] = {.vect_s16 = vsubq_s16(vreinterpretq_s16_u16(acc[j].val[3]), total_bias)};
  }

  return int16_accums;
}

template <size_t kBottomLevelBatchSize, size_t kNumQueries>
SCANN_SSE4_INLINE array<const uint8_t*, kBottomLevelBatchSize>
MakeBottomLevelBatchLookupArray(
    array<const uint8_t*, kNumQueries> mid_level_lookups, size_t start) {
  DCHECK_LE(start + kBottomLevelBatchSize, kNumQueries);
  array<const uint8_t*, kBottomLevelBatchSize> result;
  for (size_t j : Seq(kBottomLevelBatchSize)) {
    result[j] = mid_level_lookups[start + j];
  }
  return result;
}

template <size_t kNumQueries, PrefetchStrategy kPrefetch>
SCANN_SSE4_INLINE Sse4<int16_t, kNumQueries, 4> Sse4LUT16MiddleLoop(
    const uint8_t* data_start, array<const uint8_t*, kNumQueries> lookup_starts,
    const DimensionIndex num_blocks) {
  constexpr size_t kSizeB = (kNumQueries == 1) ? 1 : 2;
  constexpr size_t kNumBCases[] = {0, 2, 1};
  constexpr size_t kNumB = (kNumQueries == 1) ? 1 : kNumBCases[kNumQueries % 3];

  constexpr size_t kRemaining = kNumQueries - kNumB * kSizeB;
  static_assert(kRemaining % 3 == 0, "");

  constexpr size_t kSizeA = 3;
  constexpr size_t kNumA = kRemaining / 3;

  Sse4<int16_t, kNumQueries, 4> result;
  for (size_t j : Seq(kNumA)) {
    const size_t start = j * kSizeA;
    auto bottom_level_lookups =
        MakeBottomLevelBatchLookupArray<kSizeA>(lookup_starts, start);
    auto acc = Sse4LUT16BottomLoop<kSizeA, kPrefetch>(
        data_start, bottom_level_lookups, num_blocks);
    for (size_t jj : Seq(kSizeA)) {
      result[start + jj] = acc[jj];
    }
  }

  for (size_t j : Seq(kNumB)) {
    const size_t start = kNumA * kSizeA + j * kSizeB;
    auto bottom_level_lookups =
        MakeBottomLevelBatchLookupArray<kSizeB>(lookup_starts, start);
    auto acc = Sse4LUT16BottomLoop<kSizeB, kPrefetch>(
        data_start, bottom_level_lookups, num_blocks);
    for (size_t jj : Seq(kSizeB)) {
      result[start + jj] = acc[jj];
    }
  }
  return result;
}

template <size_t kNumQueries, PrefetchStrategy kPrefetch>
SCANN_SSE4_INLINE Sse4<int32_t, kNumQueries, 8> Sse4LUT16BottomLoopInt32(
    const uint8_t* data_start, array<const uint8_t*, kNumQueries> lookup_starts,
    DimensionIndex num_blocks) {
  Sse4<int32_t, kNumQueries, 8> int32_accums = sse4::Zeros();
  for (DimensionIndex k = 0; k < num_blocks;) {
    DimensionIndex reaccumulate_limit = std::min(num_blocks - k, uint64_t{256});
    k += reaccumulate_limit;
    auto int16_accums = Sse4LUT16MiddleLoop<kNumQueries, kPrefetch>(
        data_start, lookup_starts, reaccumulate_limit);
    data_start += 16 * reaccumulate_limit;
    for (size_t j : Seq(kNumQueries)) {
      int32_accums[j] += int16_accums[j].template ExpandTo<int32_t>();
      lookup_starts[j] += 16 * reaccumulate_limit;
    }
  }
  return int32_accums;
}

}  // namespace

template <size_t kNumQueries, PrefetchStrategy kPrefetch>
SCANN_SSE4_OUTLINE void LUT16Sse4<kNumQueries, kPrefetch>::GetInt16Distances(
    LUT16Args<int16_t> args) {
  const uint8_t* packed_dataset = args.packed_dataset;
  const size_t num_32dp_simd_iters = args.num_32dp_simd_iters;
  const size_t num_blocks = args.num_blocks;
  auto lookups = ToLocalArray<kNumQueries>(args.lookups);
  auto distances = ToLocalArray<kNumQueries>(args.distances);
  for (size_t k : Seq(num_32dp_simd_iters)) {
    const size_t dp_idx = k * 32;

    const uint8_t* data_start = packed_dataset + dp_idx * num_blocks / 2;
    auto int16_accums = Sse4LUT16MiddleLoop<kNumQueries, kPrefetch>(
        data_start, lookups, num_blocks);
    for (size_t j : Seq(kNumQueries)) {
      int16_accums[j].Store(distances[j] + dp_idx);
    }
  }
}

template <size_t kNumQueries, PrefetchStrategy kPrefetch>
SCANN_SSE4_OUTLINE void LUT16Sse4<kNumQueries, kPrefetch>::GetInt32Distances(
    LUT16Args<int32_t> args) {
  const uint8_t* packed_dataset = args.packed_dataset;
  const size_t num_32dp_simd_iters = args.num_32dp_simd_iters;
  const size_t num_blocks = args.num_blocks;
  auto lookups = ToLocalArray<kNumQueries>(args.lookups);
  auto distances = ToLocalArray<kNumQueries>(args.distances);
  for (DatapointIndex k = 0; k < num_32dp_simd_iters; k++) {
    const uint8_t* data_start = packed_dataset + k * 16 * num_blocks;
    Sse4For<int32_t, kNumQueries, 32> int32_accums =
        Sse4LUT16BottomLoopInt32<kNumQueries, kPrefetch>(data_start, lookups,
                                                         num_blocks);
    for (size_t j : Seq(kNumQueries)) {
      int32_accums[j].Store(distances[j] + 32 * k);
    }
  }
}

template <size_t kNumQueries, PrefetchStrategy kPrefetch>
SCANN_SSE4_OUTLINE void LUT16Sse4<kNumQueries, kPrefetch>::GetFloatDistances(
    LUT16Args<float> args, ConstSpan<float> inv_fp_multipliers) {
  const uint8_t* packed_dataset = args.packed_dataset;
  const size_t num_32dp_simd_iters = args.num_32dp_simd_iters;
  const size_t num_blocks = args.num_blocks;
  auto lookups = ToLocalArray<kNumQueries>(args.lookups);
  auto distances = ToLocalArray<kNumQueries>(args.distances);
  auto mults = ToLocalArray<kNumQueries>(inv_fp_multipliers);

  for (DatapointIndex k = 0; k < num_32dp_simd_iters; k++) {
    const uint8_t* data_start = packed_dataset + k * 16 * num_blocks;
    auto int32_accums = Sse4LUT16BottomLoopInt32<kNumQueries, kPrefetch>(
        data_start, lookups, num_blocks);
    for (size_t j : Seq(kNumQueries)) {
      (int32_accums[j].template ConvertTo<float>() * Sse4<float>(mults[j]))
          .Store(distances[j] + 32 * k);
    }
  }
}

namespace {
template <size_t kNumQueries, PrefetchStrategy kPrefetch, typename TopN>
SCANN_SSE4_INLINE void GetTopInt16DistancesImpl(
    LUT16ArgsTopN<int16_t, TopN> args) {
  const uint8_t* packed_dataset = args.packed_dataset;
  const size_t num_32dp_simd_iters = args.num_32dp_simd_iters;
  const size_t num_blocks = args.num_blocks;
  auto lookups = ToLocalArray<kNumQueries>(args.lookups);
  const DatapointIndex first_dp_index = args.first_dp_index;
  const uint32_t final_mask = GetFinalMask32(args.num_datapoints);
  DCHECK_EQ(num_32dp_simd_iters, DivRoundUp(args.num_datapoints, 32));

  Sse4<int16_t> simd_thresholds[kNumQueries];
  for (size_t j : Seq(kNumQueries)) {
    const int16_t int16_threshold = args.fast_topns[j]->epsilon();
    simd_thresholds[j] = int16_threshold;
  }

  typename TopN::Mutator topn_mutators[kNumQueries];
  for (size_t j : Seq(kNumQueries)) {
    args.fast_topns[j]->AcquireMutator(&topn_mutators[j]);
  }

  int16_t distances_buffer[32];
  auto restrict_whitelist_ptrs =
      args.template GetRestrictWhitelistPtrs<kNumQueries>();
  for (DatapointIndex k : Seq(num_32dp_simd_iters)) {
    bool can_skip_batch = true;
    for (size_t j : Seq(kNumQueries)) {
      if (restrict_whitelist_ptrs[j] == nullptr ||
          restrict_whitelist_ptrs[j][k] != 0) {
        can_skip_batch = false;
        break;
      }
    }
    if (can_skip_batch) continue;

    const uint8_t* data_start = packed_dataset + k * 16 * num_blocks;
    auto int16_accums = Sse4LUT16MiddleLoop<kNumQueries, kPrefetch>(
        data_start, lookups, num_blocks);
    for (size_t j : Seq(kNumQueries)) {
      auto compute_push_mask = [&]() SCANN_INLINE_LAMBDA {
        return hw_alg::GetComparisonMask((*int16_accums[j][0]).vect_s16,(*int16_accums[j][1]).vect_s16,
            (*int16_accums[j][2]).vect_s16,(*int16_accums[j][3]).vect_s16,(*simd_thresholds[j]).vect_s16);
      };
      uint32_t push_mask = compute_push_mask();

      if (!push_mask) continue;

      int16x8x4_t dists;
      dists.val[0] = (*int16_accums[j][0]).vect_s16;
      dists.val[1] = (*int16_accums[j][1]).vect_s16;
      dists.val[2] = (*int16_accums[j][2]).vect_s16;
      dists.val[3] = (*int16_accums[j][3]).vect_s16;
      vst4q_s16(distances_buffer, dists);

      if (k == num_32dp_simd_iters - 1) {
        push_mask &= final_mask;
      }
      if (restrict_whitelist_ptrs[j]) {
        push_mask &= restrict_whitelist_ptrs[j][k];
      }

      while (push_mask) {
        const int offset = bits::FindLSBSetNonZero(push_mask);
        push_mask &= (push_mask - 1);
        const DatapointIndex dp_idx = first_dp_index + 32 * k + offset;
        DCHECK(
            !restrict_whitelist_ptrs[j] ||
            args.restrict_whitelists[j].IsWhitelisted(dp_idx - first_dp_index))
            << dp_idx;
        const int16_t distance = distances_buffer[offset];
        const bool needs_collection = topn_mutators[j].Push(dp_idx, distance);
        if (ABSL_PREDICT_FALSE(needs_collection)) {
          topn_mutators[j].GarbageCollect();

          simd_thresholds[j] = topn_mutators[j].epsilon();

          push_mask &= compute_push_mask();
        }
      }
    }
  }
}
}  // namespace

template <size_t kNumQueries, PrefetchStrategy kPrefetch>
SCANN_SSE4_OUTLINE void LUT16Sse4<kNumQueries, kPrefetch>::GetTopInt16Distances(
    LUT16ArgsTopN<int16_t> args) {
  return GetTopInt16DistancesImpl<kNumQueries, kPrefetch>(std::move(args));
}

SCANN_SSE4_INLINE int16_t GetInt16Threshold(float float_threshold) {
  constexpr float kMaxValue = numeric_limits<int16_t>::max();

  return std::min(float_threshold, kMaxValue);
}

namespace {
template <size_t kNumQueries, PrefetchStrategy kPrefetch,
          bool kWithSpilling = false, typename TopN>
SCANN_SSE4_INLINE void GetTopFloatDistancesImpl(
    LUT16ArgsTopN<float, TopN> args) {
  if constexpr (kWithSpilling) {
    if (args.datapoint_translation_predicate)
      LOG(FATAL)
          << "Spilling is incompatible with datapoint translation predicates.";
  }

  const uint8_t* packed_dataset = args.packed_dataset;
  const size_t num_32dp_simd_iters = args.num_32dp_simd_iters;
  const size_t num_blocks = args.num_blocks;
  auto lookups = ToLocalArray<kNumQueries>(args.lookups);
  const DatapointIndex first_dp_index = args.first_dp_index;
  const uint32_t final_mask = GetFinalMask32(args.num_datapoints);
  DCHECK_EQ(num_32dp_simd_iters, DivRoundUp(args.num_datapoints, 32));

  auto biases = ToLocalArray<kNumQueries>(args.biases);
  Sse4<float> simd_biases[kNumQueries];
  for (size_t j : Seq(kNumQueries)) {
    simd_biases[j] = biases[j];
  }

  auto mults = ToLocalArray<kNumQueries>(args.fixed_point_multipliers);
  Sse4<float> inv_mults[kNumQueries];
  for (size_t j : Seq(kNumQueries)) {
    inv_mults[j] = 1.0 / mults[j];
  }

  Sse4<int16_t> simd_thresholds[kNumQueries];
  for (size_t j : Seq(kNumQueries)) {
    const float epsilon = args.fast_topns[j]->epsilon();
    const float float_threshold = (epsilon - biases[j]) * mults[j];
    const int16_t int16_threshold = GetInt16Threshold(float_threshold);
    simd_thresholds[j] = int16_threshold;
  }

  typename TopN::Mutator topn_mutators[kNumQueries];
  for (size_t j : Seq(kNumQueries)) {
    args.fast_topns[j]->AcquireMutator(&topn_mutators[j]);
  }

  float distances_buffer[32];
  auto restrict_whitelist_ptrs =
      args.template GetRestrictWhitelistPtrs<kNumQueries>();
  for (DatapointIndex k : Seq(num_32dp_simd_iters)) {
    bool can_skip_batch = true;
    for (size_t j : Seq(kNumQueries)) {
      if (restrict_whitelist_ptrs[j] == nullptr ||
          restrict_whitelist_ptrs[j][k] != 0) {
        can_skip_batch = false;
        break;
      }
    }
    if (can_skip_batch) continue;

    const uint8_t* data_start = packed_dataset + k * 16 * num_blocks;
    auto int16_accums = Sse4LUT16MiddleLoop<kNumQueries, kPrefetch>(
        data_start, lookups, num_blocks);
    for (size_t j : Seq(kNumQueries)) {
      auto compute_push_mask = [&]() SCANN_INLINE_LAMBDA {
        return hw_alg::GetComparisonMask((*int16_accums[j][0]).vect_s16,(*int16_accums[j][1]).vect_s16,
            (*int16_accums[j][2]).vect_s16,(*int16_accums[j][3]).vect_s16,(*simd_thresholds[j]).vect_s16);
      };
      uint32_t push_mask = compute_push_mask();

      if (!push_mask) continue;

      auto fvals = int16_accums[j]
                       .template ExpandTo<int32_t>()
                       .template ConvertTo<float>();
      fvals = (fvals * inv_mults[j] + simd_biases[j]);
      float32x4x4_t dists1, dists2;
      dists1.val[0] = (*fvals[0]).vect_f32;
      dists1.val[1] = (*fvals[2]).vect_f32;
      dists1.val[2] = (*fvals[4]).vect_f32;
      dists1.val[3] = (*fvals[6]).vect_f32;
      dists2.val[0] = (*fvals[1]).vect_f32;
      dists2.val[1] = (*fvals[3]).vect_f32;
      dists2.val[2] = (*fvals[5]).vect_f32;
      dists2.val[3] = (*fvals[7]).vect_f32;
      vst4q_f32(distances_buffer, dists1);
      vst4q_f32(distances_buffer + 16, dists2);

      if (k == num_32dp_simd_iters - 1) {
        push_mask &= final_mask;
      }
      if (restrict_whitelist_ptrs[j]) {
        push_mask &= restrict_whitelist_ptrs[j][k];
      }
      if (args.batch_filter_predicate) {
        push_mask =
            args.batch_filter_predicate(first_dp_index + 32 * k, push_mask,
                                        args.datapoint_translation_predicate);
      }

      while (push_mask) {
        const int offset = bits::FindLSBSetNonZero(push_mask);
        push_mask &= (push_mask - 1);
        DatapointIndex dp_idx;
        if constexpr (!kWithSpilling) {
          dp_idx = first_dp_index + 32 * k + offset;
        }
        DCHECK(
            !restrict_whitelist_ptrs[j] ||
            args.restrict_whitelists[j].IsWhitelisted(dp_idx - first_dp_index))
            << dp_idx;
        if (args.datapoint_translation_predicate) {
          dp_idx = args.datapoint_translation_predicate(dp_idx);
        }

        // DCHECK_LE(distances_buffer[offset],
        //           topn_mutators[j].epsilon() + (*inv_mults[j])[0]);
        const bool needs_gc = topn_mutators[j].PushNoEpsilonCheck(
            dp_idx, distances_buffer[offset]);
        if (ABSL_PREDICT_FALSE(needs_gc)) {
          topn_mutators[j].GarbageCollect();

          const float new_epsilon = topn_mutators[j].epsilon();
          const float float_threshold = (new_epsilon - biases[j]) * mults[j];
          const int16_t int16_threshold = GetInt16Threshold(float_threshold);
          simd_thresholds[j] = int16_threshold;

          push_mask &= compute_push_mask();
        }
      }
    }
  }
}
}  // namespace

template <size_t kNumQueries, PrefetchStrategy kPrefetch>
SCANN_SSE4_OUTLINE void LUT16Sse4<kNumQueries, kPrefetch>::GetTopFloatDistances(
    LUT16ArgsTopN<float> args) {
  return GetTopFloatDistancesImpl<kNumQueries, kPrefetch>(std::move(args));
}

}  // namespace asymmetric_hashing_internal
}  // namespace research_scann

#endif
