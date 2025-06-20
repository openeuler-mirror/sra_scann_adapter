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

#ifndef SCANN_UTILS_INTRINSICS_AVX1_H_
#define SCANN_UTILS_INTRINSICS_AVX1_H_

#include <algorithm>
#include <cstdint>
#include <utility>

#include "scann/utils/index_sequence.h"
#include "scann/utils/intrinsics/attributes.h"
#include "scann/utils/intrinsics/flags.h"
#include "scann/utils/intrinsics/sse4.h"
#include "scann/utils/types.h"

#ifdef __aarch64__

#include "avx2ki.h"

namespace research_scann {
namespace avx1 {

static constexpr PlatformGeneration kPlatformGeneration = kSandyBridgeAvx1;

template <typename T, size_t kNumElementsRequired>
constexpr size_t InferNumRegisters() {
  constexpr size_t kRegisterBytes = IsFloatingType<T>() ? 32 : 16;
  constexpr size_t kElementsPerRegister = kRegisterBytes / sizeof(T);

  static_assert(kNumElementsRequired > 0);
  static_assert(IsDivisibleBy(kNumElementsRequired, kElementsPerRegister));

  return kNumElementsRequired / kElementsPerRegister;
}

}  // namespace avx1

template <typename T, size_t kNumRegisters = 1, size_t... kTensorNumRegisters>
class Avx1;

struct Avx1Zeros {};
struct Avx1Uninitialized {};

template <typename T, size_t kNumRegistersInferred>
class Avx1<T, kNumRegistersInferred> {
 public:
  SCANN_DECLARE_COPYABLE_CLASS(Avx1);
  static_assert(IsSameAny<T, float, double, int8_t, int16_t, int32_t, int64_t,
                          uint8_t, uint16_t, uint32_t, uint64_t>());

  static constexpr size_t kRegisterBits = IsFloatingType<T>() ? 256 : 128;
  static constexpr size_t kRegisterBytes = IsFloatingType<T>() ? 32 : 16;
  static constexpr size_t kNumRegisters = kNumRegistersInferred;
  static constexpr size_t kElementsPerRegister = kRegisterBytes / sizeof(T);
  static constexpr size_t kNumElements = kNumRegisters * kElementsPerRegister;

  static auto InferIntelType() {
    if constexpr (std::is_same_v<T, float>) {
      return __m256();
    } else if constexpr (std::is_same_v<T, double>) {
      return __m256d();
    } else {
      return __m128i();
    }
  }
  using IntelType = decltype(InferIntelType());
  static_assert(sizeof(IntelType) == kRegisterBytes);

  Avx1(Avx1Uninitialized) {}
  Avx1() : Avx1(Avx1Uninitialized()) {}

  SCANN_AVX1_INLINE Avx1(Avx1Zeros) { Clear(); }

  SCANN_AVX1_INLINE Avx1(IntelType val) {
    static_assert(kNumRegisters == 1);
    *this = val;
  }

  SCANN_AVX1_INLINE Avx1(T val) { *this = Broadcast(val); }

  template <typename U, size_t kOther>
  SCANN_AVX1_INLINE explicit Avx1(const Avx1<U, kOther>& other) {
    Avx1& me = *this;
    for (size_t j : Seq(kNumRegisters)) {
      if constexpr (kOther == kNumRegisters) {
        me[j] = *other[j];
      } else if constexpr (kOther == 1) {
        me[j] = *other[0];
      } else {
        static_assert(kOther == kNumRegisters || kOther == 1);
      }
    }
  }

  SCANN_AVX1_INLINE Avx1& operator=(Avx1Zeros val) {
    Clear();
    return *this;
  }

  SCANN_AVX1_INLINE Avx1& operator=(IntelType val) {
    static_assert(kNumRegisters == 1,
                  "To intentionally perform register-wise broadcast, "
                  "explicitly cast to an Avx1<T>");
    registers_[0] = val;
    return *this;
  }

  SCANN_AVX1_INLINE Avx1& operator=(T val) {
    *this = Broadcast(val);
    return *this;
  }

  SCANN_AVX1_INLINE IntelType operator*() const {
    static_assert(kNumRegisters == 1);
    return registers_[0];
  }

  SCANN_AVX1_INLINE Avx1<T, 1>& operator[](size_t idx) {
    if constexpr (kNumRegisters == 1) {
      DCHECK_EQ(idx, 0);
      return *this;
    } else {
      DCHECK_LT(idx, kNumRegisters);
      return registers_[idx];
    }
  }

  SCANN_AVX1_INLINE const Avx1<T, 1>& operator[](size_t idx) const {
    if constexpr (kNumRegisters == 1) {
      DCHECK_EQ(idx, 0);
      return *this;
    } else {
      DCHECK_LT(idx, kNumRegisters);
      return registers_[idx];
    }
  }

  static SCANN_AVX1_INLINE IntelType ZeroOneRegister() {
    if constexpr (IsSameAny<T, float>()) {
      return _mm256_setzero_ps();
    } else if constexpr (IsSameAny<T, double>()) {
      return _mm256_setzero_pd();
    } else {
      return _mm_setzero_si128();
    }
  }

  static SCANN_AVX1_INLINE Avx1 Zeros() {
    Avx1<T, kNumRegisters> ret;
    for (size_t j : Seq(kNumRegisters)) {
      ret[j] = ZeroOneRegister();
    }
    return ret;
  }

  SCANN_AVX1_INLINE Avx1& Clear() {
    for (size_t j : Seq(kNumRegisters)) {
      registers_[j] = ZeroOneRegister();
    }
    return *this;
  }

  static SCANN_AVX1_INLINE IntelType BroadcastOneRegister(T x) {
    if constexpr (IsSameAny<T, float>()) {
      return _mm256_set1_ps(x);
    }
    if constexpr (IsSameAny<T, double>()) {
      return _mm256_set1_pd(x);
    }

    if constexpr (IsSameAny<T, int8_t, uint8_t>()) {
      return _mm_set1_epi8(x);
    }
    if constexpr (IsSameAny<T, int16_t, uint16_t>()) {
      return _mm_set1_epi16(x);
    }
    if constexpr (IsSameAny<T, int32_t, uint32_t>()) {
      return _mm_set1_epi32(x);
    }
    if constexpr (IsSameAny<T, int64_t, uint64_t>()) {
      return _mm_set1_epi64(__m64{static_cast<int64_t>(x)});
    }
    LOG(FATAL) << "Undefined";
  }

  SCANN_AVX1_INLINE static Avx1 Broadcast(T x) {
    Avx1 ret;
    for (size_t j : Seq(kNumRegisters)) {
      ret[j] = BroadcastOneRegister(x);
    }
    return ret;
  }

  template <bool kAligned = false>
  static SCANN_AVX1_INLINE IntelType LoadOneRegister(const T* address) {
    if constexpr (kAligned) {
      if constexpr (IsSameAny<T, float>()) {
        return _mm256_load_ps(address);
      } else if constexpr (IsSameAny<T, double>()) {
        return _mm256_load_pd(address);
      } else {
        return _mm_load_si128(reinterpret_cast<const __m128i*>(address));
      }
    } else {
      if constexpr (IsSameAny<T, float>()) {
        return _mm256_loadu_ps(address);
      } else if constexpr (IsSameAny<T, double>()) {
        return _mm256_loadu_pd(address);
      } else {
        return _mm_loadu_si128(reinterpret_cast<const __m128i*>(address));
      }
    }
  }

  template <bool kAligned = false>
  SCANN_AVX1_INLINE static Avx1 Load(const T* address) {
    Avx1 ret;
    for (size_t j : Seq(kNumRegisters)) {
      ret[j] = LoadOneRegister<kAligned>(address + j * kElementsPerRegister);
    }
    return ret;
  }

  static SCANN_AVX1_INLINE void StoreOneRegister(T* address, IntelType x) {
    if constexpr (IsSameAny<T, float>()) {
      _mm256_storeu_ps(address, x);
    } else if constexpr (IsSameAny<T, double>()) {
      _mm256_storeu_pd(address, x);
    } else {
      _mm_storeu_si128(reinterpret_cast<__m128i*>(address), x);
    }
  }

  SCANN_AVX1_INLINE void Store(T* address) const {
    const Avx1& me = *this;
    for (size_t j : Seq(kNumRegisters)) {
      StoreOneRegister(address + j * kElementsPerRegister, *me[j]);
    }
  }

  SCANN_AVX1_INLINE array<T, kNumElements> Store() const {
    array<T, kNumElements> ret;
    Store(ret.data());
    return ret;
  }

  template <size_t kOther, typename Op,
            size_t kOutput = std::max(kNumRegisters, kOther)>
  static SCANN_AVX1_INLINE Avx1<T, kOutput> BinaryOperatorImpl(
      const Avx1& me, const Avx1<T, kOther>& other, Op fn) {
    Avx1<T, kOutput> ret;
    for (size_t j : Seq(Avx1<T, kOutput>::kNumRegisters)) {
      if constexpr (kOther == kNumRegisters) {
        ret[j] = fn(*me[j], *other[j]);
      } else if constexpr (kNumRegisters == 1) {
        ret[j] = fn(*me[0], *other[j]);
      } else if constexpr (kOther == 1) {
        ret[j] = fn(*me[j], *other[0]);
      } else {
        static_assert(kOther == kNumRegisters || kNumRegisters == 1 ||
                      kOther == 1);
      }
    }
    return ret;
  }

  static SCANN_AVX1_INLINE IntelType Add(IntelType a, IntelType b) {
    if constexpr (IsSameAny<T, float>()) {
      return _mm256_add_ps(a, b);
    }
    if constexpr (IsSameAny<T, double>()) {
      return _mm256_add_pd(a, b);
    }

    if constexpr (IsSameAny<T, int8_t, uint8_t>()) {
      return _mm_add_epi8(a, b);
    }
    if constexpr (IsSameAny<T, int16_t, uint16_t>()) {
      return _mm_add_epi16(a, b);
    }
    if constexpr (IsSameAny<T, int32_t, uint32_t>()) {
      return _mm_add_epi32(a, b);
    }
    if constexpr (IsSameAny<T, int64_t, uint64_t>()) {
      return _mm_add_epi64(a, b);
    }
    LOG(FATAL) << "Undefined";
  }

  template <size_t kOther>
  SCANN_AVX1_INLINE auto operator+(const Avx1<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &Add);
  }

  static SCANN_AVX1_INLINE IntelType Subtract(IntelType a, IntelType b) {
    if constexpr (IsSameAny<T, float>()) {
      return _mm256_sub_ps(a, b);
    }
    if constexpr (IsSameAny<T, double>()) {
      return _mm256_sub_pd(a, b);
    }

    if constexpr (IsSameAny<T, int8_t, uint8_t>()) {
      return _mm_sub_epi8(a, b);
    }
    if constexpr (IsSameAny<T, int16_t, uint16_t>()) {
      return _mm_sub_epi16(a, b);
    }
    if constexpr (IsSameAny<T, int32_t, uint32_t>()) {
      return _mm_sub_epi32(a, b);
    }
    if constexpr (IsSameAny<T, int64_t, uint64_t>()) {
      return _mm_sub_epi64(a, b);
    }
  }

  template <size_t kOther>
  SCANN_AVX1_INLINE auto operator-(const Avx1<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &Subtract);
  }

  static SCANN_AVX1_INLINE IntelType Multiply(IntelType a, IntelType b) {
    if constexpr (IsSame<T, float>()) {
      return _mm256_mul_ps(a, b);
    }
    if constexpr (IsSame<T, double>()) {
      return _mm256_mul_pd(a, b);
    }

    static_assert(!IsSame<T, int8_t>(), "There's no 8-bit '*' instruction");
    if constexpr (IsSame<T, int16_t>()) {
      return _mm_mullo_epi16(a, b);
    }
    if constexpr (IsSame<T, int32_t>()) {
      return _mm_mullo_epi32(a, b);
    }
    static_assert(!IsSame<T, int64_t>(),
                  "_mm_mullo_epi64 is introduced in AVX-512");

    static_assert(!IsSameAny<T, uint8_t, uint16_t, uint32_t, uint64_t>(),
                  "Not Implemented. Unsigned multiplication is limited to "
                  "_mm_mul_epu32, which expands from uint32=>uint64.");
  }

  template <size_t kOther>
  SCANN_AVX1_INLINE auto operator*(const Avx1<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &Multiply);
  }

  static SCANN_AVX1_INLINE IntelType Divide(IntelType a, IntelType b) {
    if constexpr (IsSame<T, float>()) {
      return _mm256_div_ps(a, b);
    }
    if constexpr (IsSame<T, double>()) {
      return _mm256_div_pd(a, b);
    }

    static_assert(!IsSameAny<T, int8_t, int16_t, int32_t, int64_t>(),
                  "There's no integer '/' operations.");

    static_assert(!IsSameAny<T, uint8_t, uint16_t, uint32_t, uint64_t>(),
                  "There's no integer '/' operations.");
  }

  template <size_t kOther>
  SCANN_AVX1_INLINE auto operator/(const Avx1<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &Divide);
  }

  static SCANN_AVX1_INLINE auto BitwiseAnd(IntelType a, IntelType b) {
    if constexpr (IsSame<T, float>()) {
      return _mm256_and_ps(a, b);
    }
    if constexpr (IsSame<T, double>()) {
      return _mm256_and_pd(a, b);
    }
    if constexpr (IsSameAny<T, int8_t, uint8_t, int16_t, uint16_t, int32_t,
                            uint32_t, int64_t, uint64_t>()) {
      return _mm_and_si128(a, b);
    }
    LOG(FATAL) << "Undefined";
  }

  template <size_t kOther>
  SCANN_AVX1_INLINE auto operator&(const Avx1<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &BitwiseAnd);
  }

  static SCANN_AVX1_INLINE auto BitwiseOr(IntelType a, IntelType b) {
    if constexpr (IsSame<T, float>()) {
      return _mm256_or_ps(a, b);
    }
    if constexpr (IsSame<T, double>()) {
      return _mm256_or_pd(a, b);
    }
    if constexpr (IsSameAny<T, int8_t, uint8_t, int16_t, uint16_t, int32_t,
                            uint32_t, int64_t, uint64_t>()) {
      return _mm_or_si128(a, b);
    }
    LOG(FATAL) << "Undefined";
  }

  template <size_t kOther>
  SCANN_AVX1_INLINE auto operator|(const Avx1<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &BitwiseOr);
  }

  static SCANN_AVX1_INLINE auto BitwiseXor(IntelType a, IntelType b) {
    if constexpr (IsSame<T, float>()) {
      return _mm256_xor_ps(a, b);
    }
    if constexpr (IsSame<T, double>()) {
      return _mm256_xor_pd(a, b);
    }
    if constexpr (IsSameAny<T, int8_t, uint8_t, int16_t, uint16_t, int32_t,
                            uint32_t, int64_t, uint64_t>()) {
      return _mm_xor_si128(a, b);
    }
    LOG(FATAL) << "Undefined";
  }

  template <size_t kOther>
  SCANN_AVX1_INLINE auto operator^(const Avx1<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &BitwiseXor);
  }

  static SCANN_AVX1_INLINE IntelType ShiftRight(IntelType x, int count) {
    static_assert(!IsSameAny<T, int8_t, uint8_t>(),
                  "There's no 8-bit '>>' instruction");
    static_assert(!IsSameAny<T, float, double>(),
                  "Bit shifting isn't defined for floating-point types.");

    if constexpr (IsSame<T, int16_t>()) {
      return _mm_srai_epi16(x, count);
    }
    if constexpr (IsSame<T, int32_t>()) {
      return _mm_srai_epi32(x, count);
    }
    if constexpr (IsSame<T, int64_t>()) {
      return _mm_srai_epi64(x, count);
    }

    if constexpr (IsSameAny<T, uint16_t>()) {
      return _mm_srli_epi16(x, count);
    }
    if constexpr (IsSame<T, uint32_t>()) {
      return _mm_srli_epi32(x, count);
    }
    if constexpr (IsSame<T, uint64_t>()) {
      return _mm_srli_epi64(x, count);
    }
    LOG(FATAL) << "Undefined";
  }

  SCANN_AVX1_INLINE Avx1 operator>>(int count) const {
    const Avx1& me = *this;
    Avx1 ret;
    for (size_t j : Seq(kNumRegisters)) {
      ret[j] = ShiftRight(*me[j], count);
    }
    return ret;
  }

  static SCANN_AVX1_INLINE IntelType ShiftLeft(IntelType x, int count) {
    static_assert(!IsSameAny<T, int8_t, uint8_t>(),
                  "There's no 8-bit '<<' instruction");
    static_assert(!IsSameAny<T, float, double>(),
                  "Bit shifting isn't defined for floating-point types.");

    if constexpr (IsSameAny<T, int16_t, uint16_t>()) {
      return _mm_slli_epi16(x, count);
    }
    if constexpr (IsSameAny<T, int32_t, uint32_t>()) {
      return _mm_slli_epi32(x, count);
    }
    if constexpr (IsSameAny<T, int64_t, uint64_t>()) {
      return _mm_slli_epi64(x, count);
    }
    LOG(FATAL) << "Undefined";
  }

  SCANN_AVX1_INLINE Avx1 operator<<(int count) const {
    const Avx1& me = *this;
    Avx1 ret;
    for (size_t j : Seq(kNumRegisters)) {
      ret[j] = ShiftLeft(*me[j], count);
    }
    return ret;
  }

  template <size_t kOther, typename Op>
  SCANN_AVX1_INLINE Avx1& AccumulateOperatorImpl(const Avx1<T, kOther>& other,
                                                 Op fn) {
    Avx1& me = *this;
    for (size_t j : Seq(kNumRegisters)) {
      if constexpr (kOther == kNumRegisters) {
        me[j] = fn(*me[j], *other[j]);
      } else if constexpr (kOther == 1) {
        me[j] = fn(*me[j], *other[0]);
      } else {
        static_assert(kOther == kNumRegisters || kOther == 1);
      }
    }
    return *this;
  }

  template <size_t kOther>
  SCANN_AVX1_INLINE Avx1& operator+=(const Avx1<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &Add);
  }

  template <size_t kOther>
  SCANN_AVX1_INLINE Avx1& operator-=(const Avx1<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &Subtract);
  }

  template <size_t kOther>
  SCANN_AVX1_INLINE Avx1& operator*=(const Avx1<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &Multiply);
  }

  template <size_t kOther>
  SCANN_AVX1_INLINE Avx1& operator/=(const Avx1<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &Divide);
  }

  template <size_t kOther>
  SCANN_AVX1_INLINE Avx1& operator&=(const Avx1<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &BitwiseAnd);
  }

  template <size_t kOther>
  SCANN_AVX1_INLINE Avx1& operator|=(const Avx1<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &BitwiseOr);
  }

  template <size_t kOther>
  SCANN_AVX1_INLINE Avx1& operator^=(const Avx1<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &BitwiseXor);
  }

  SCANN_AVX1_INLINE Avx1& operator<<=(int count) {
    Avx1& me = *this;
    for (size_t j : Seq(kNumRegisters)) {
      me[j] = ShiftLeft(*me[j], count);
    }
    return *this;
  }

  SCANN_AVX1_INLINE Avx1& operator>>=(int count) {
    Avx1& me = *this;
    for (size_t j : Seq(kNumRegisters)) {
      me[j] = ShiftRight(*me[j], count);
    }
    return *this;
  }

  template <size_t kOther = kNumRegisters, typename Op>
  SCANN_AVX1_INLINE auto ComparisonOperatorImpl(const Avx1& me,
                                                const Avx1<T, kOther>& other,
                                                Op fn) const {
    Avx1<T, std::max(kNumRegisters, kOther)> masks;
    for (size_t j : Seq(std::max(kNumRegisters, kOther))) {
      if constexpr (kOther == kNumRegisters) {
        masks[j] = fn(*me[j], *other[j]);
      } else if constexpr (kNumRegisters == 1) {
        masks[j] = fn(*me[0], *other[j]);
      } else if constexpr (kOther == 1) {
        masks[j] = fn(*me[j], *other[0]);
      } else {
        static_assert(kOther == kNumRegisters || kNumRegisters == 1 ||
                      kOther == 1);
      }
    }
    return masks;
  }

  static SCANN_AVX1_INLINE IntelType LessThan(IntelType a, IntelType b) {
    if constexpr (IsSameAny<T, float>()) {
      return _mm256_cmp_ps(a, b, _CMP_LT_OS);
    }
    if constexpr (IsSameAny<T, double>()) {
      return _mm256_cmp_pd(a, b, _CMP_LT_OS);
    }

    if constexpr (IsSameAny<T, int8_t, int16_t, int32_t, int64_t>()) {
      return GreaterThan(b, a);
    }

    static_assert(!IsSameAny<T, uint8_t, uint16_t, uint32_t, uint64_t>(),
                  "Prior to AVX-512, there are no unsigned comparison ops");
    LOG(FATAL) << "Undefined";
  }

  template <size_t kOther = kNumRegisters>
  SCANN_AVX1_INLINE auto operator<(const Avx1<T, kOther>& other) const {
    return ComparisonOperatorImpl(*this, other, &LessThan);
  }

  static SCANN_AVX1_INLINE IntelType LessOrEquals(IntelType a, IntelType b) {
    if constexpr (IsSameAny<T, float>()) {
      return _mm256_cmp_ps(a, b, _CMP_LE_OS);
    }
    if constexpr (IsSameAny<T, double>()) {
      return _mm256_cmp_pd(a, b, _CMP_LE_OS);
    }

    static_assert(!IsSameAny<T, int8_t, uint8_t, int16_t, uint16_t, int32_t,
                             uint32_t, int64_t, uint64_t>(),
                  "Prior to AVX-512, the only integer comparison ops are '<', "
                  "'>', and '==' for signed integers.");
    LOG(FATAL) << "Undefined";
  }

  template <size_t kOther = kNumRegisters>
  SCANN_AVX1_INLINE auto operator<=(const Avx1<T, kOther>& other) const {
    return ComparisonOperatorImpl(*this, other, &LessOrEquals);
  }

  static SCANN_AVX1_INLINE IntelType Equals(IntelType a, IntelType b) {
    if constexpr (IsSameAny<T, float>()) {
      return _mm256_cmp_ps(a, b, _CMP_EQ_OS);
    }
    if constexpr (IsSameAny<T, double>()) {
      return _mm256_cmp_pd(a, b, _CMP_EQ_OS);
    }

    if constexpr (IsSameAny<T, int8_t, uint8_t>()) {
      return _mm_cmpeq_epi8(a, b);
    }
    if constexpr (IsSameAny<T, int16_t, uint16_t>()) {
      return _mm_cmpeq_epi16(a, b);
    }
    if constexpr (IsSameAny<T, int32_t, uint32_t>()) {
      return _mm_cmpeq_epi32(a, b);
    }
    if constexpr (IsSameAny<T, int64_t, uint64_t>()) {
      return _mm_cmpeq_epi64(a, b);
    }
    LOG(FATAL) << "Undefined";
  }

  template <size_t kOther>
  SCANN_AVX1_INLINE auto operator==(const Avx1<T, kOther>& other) const {
    return ComparisonOperatorImpl(*this, other, &Equals);
  }

  static SCANN_AVX1_INLINE IntelType GreaterOrEquals(IntelType a, IntelType b) {
    if constexpr (IsSameAny<T, float>()) {
      return _mm256_cmp_ps(a, b, _CMP_GE_OS);
    }
    if constexpr (IsSameAny<T, double>()) {
      return _mm256_cmp_pd(a, b, _CMP_GE_OS);
    }

    static_assert(!IsSameAny<T, int8_t, uint8_t, int16_t, uint16_t, int32_t,
                             uint32_t, int64_t, uint64_t>(),
                  "Prior to AVX-512, the only integer comparison ops are '<', "
                  "'>', and '==' for signed integers.");
    LOG(FATAL) << "Undefined";
  }

  template <size_t kOther = kNumRegisters>
  SCANN_AVX1_INLINE auto operator>=(const Avx1<T, kOther>& other) const {
    return ComparisonOperatorImpl(*this, other, &GreaterOrEquals);
  }

  static SCANN_AVX1_INLINE IntelType GreaterThan(IntelType a, IntelType b) {
    if constexpr (IsSameAny<T, float>()) {
      return _mm256_cmp_ps(a, b, _CMP_GT_OS);
    }
    if constexpr (IsSameAny<T, double>()) {
      return _mm256_cmp_pd(a, b, _CMP_GT_OS);
    }

    if constexpr (IsSameAny<T, int8_t>()) {
      return _mm_cmpgt_epi8(a, b);
    }
    if constexpr (IsSameAny<T, int16_t>()) {
      return _mm_cmpgt_epi16(a, b);
    }
    if constexpr (IsSameAny<T, int32_t>()) {
      return _mm_cmpgt_epi32(a, b);
    }
    if constexpr (IsSameAny<T, int64_t>()) {
      return _mm_cmpgt_epi64(a, b);
    }

    static_assert(!IsSameAny<T, uint8_t, uint16_t, uint32_t, uint64_t>(),
                  "Prior to AVX-512, there are no unsigned comparison ops");
    LOG(FATAL) << "Undefined";
  }

  template <size_t kOther = kNumRegisters>
  SCANN_AVX1_INLINE auto operator>(const Avx1<T, kOther>& other) const {
    return ComparisonOperatorImpl(*this, other, &GreaterThan);
  }

  SCANN_AVX1_INLINE uint32_t MaskFromHighBits() const {
    static_assert(kNumRegisters == 1);
    const auto& me = *this;

    if constexpr (IsSame<T, float>()) {
      return _mm256_movemask_ps(*me[0]);
    }
    if constexpr (IsSame<T, double>()) {
      return _mm256_movemask_pd(*me[0]);
    }

    if constexpr (IsSameAny<T, int8_t, uint8_t>()) {
      return _mm_movemask_epi8(*me[0]);
    }
    static_assert(!IsSameAny<T, int16_t, uint16_t>(),
                  "There's no efficient single-register equivalent to the "
                  "missing _mm_movemask_epi16 op. Try the two register "
                  "MaskFromHighBits helper method.");
    if constexpr (IsSameAny<T, int32_t, uint32_t>()) {
      return _mm_movemask_ps(_mm_castsi128_ps(*me[0]));
    }
    if constexpr (IsSameAny<T, int64_t, uint64_t>()) {
      return _mm_movemask_pd(_mm_castsi128_pd(*me[0]));
    }
  }

  SCANN_AVX1_INLINE T GetLowElement() const {
    static_assert(kNumRegisters == 1);
    const auto& me = *this;

    if constexpr (IsSameAny<T, float, double>()) {
      return (*me[0])[0];
    }

    if constexpr (IsSameAny<T, int8_t, uint8_t>()) {
      return _mm_extract_epi8(*me[0], 0);
    }
    if constexpr (IsSameAny<T, int16_t, uint16_t>()) {
      return _mm_extract_epi16(*me[0], 0);
    }
    if constexpr (IsSameAny<T, int32_t, uint32_t>()) {
      return _mm_cvtsi128_si32(*me[0]);
    }
    if constexpr (IsSameAny<T, int64_t, uint64_t>()) {
      return (*me[0])[0];
    }
    LOG(FATAL) << "Undefined";
  }

  template <typename U>
  static SCANN_AVX1_INLINE typename Avx1<U>::IntelType ConvertTwoRegisters(
      IntelType lo, IntelType hi) {
    constexpr int kAABB = 0x20;
    const __m256i concat = _mm256_permute2f128_si256(
        _mm256_zextsi128_si256(lo), _mm256_zextsi128_si256(hi), kAABB);

    if constexpr (IsSame<T, int32_t>() && IsSame<U, float>()) {
      return _mm256_cvtepi32_ps(concat);
    }
    if constexpr (IsSame<T, int64_t>() && IsSame<U, double>()) {
      return _mm256_cvtepi64_pd(concat);
    }
    LOG(FATAL) << "Undefined";
  }

  template <typename U>
  SCANN_AVX1_INLINE Avx1<U, kNumRegisters / 2> ConvertTo() const {
    static_assert(IsDivisibleBy(kNumRegisters, 2));
    const Avx1& me = *this;
    Avx1<U, kNumRegisters / 2> ret;
    for (size_t j : Seq(kNumRegisters / 2)) {
      ret[j] = ConvertTwoRegisters<U>(*me[j * 2], *me[j * 2 + 1]);
    }
    return ret;
  }

  static SCANN_AVX1_INLINE auto InferExpansionType() {
    if constexpr (IsSame<T, float>()) {
      return double();
    }
    if constexpr (IsSame<T, double>()) {
      return double();
    }

    if constexpr (IsSame<T, int8_t>()) {
      return int16_t();
    }
    if constexpr (IsSame<T, int16_t>()) {
      return int32_t();
    }
    if constexpr (IsSame<T, int32_t>()) {
      return int64_t();
    }
    if constexpr (IsSame<T, int64_t>()) {
      return int64_t();
    }

    if constexpr (IsSame<T, uint8_t>()) {
      return uint16_t();
    }
    if constexpr (IsSame<T, uint16_t>()) {
      return uint32_t();
    }
    if constexpr (IsSame<T, uint32_t>()) {
      return uint64_t();
    }
    if constexpr (IsSame<T, uint64_t>()) {
      return uint64_t();
    }
  }
  using ExpansionType = decltype(InferExpansionType());
  using ExpansionIntelType = typename Avx1<ExpansionType>::IntelType;
  using ExpandsTo = Avx1<ExpansionType, 2 * kNumRegisters>;

  static SCANN_AVX1_INLINE pair<ExpansionIntelType, ExpansionIntelType>
  ExpandOneRegister(IntelType x) {
    if constexpr (IsSame<T, float>()) {
      __m128 hi = _mm256_extractf128_ps(x, 1);
      __m128 lo = _mm256_castps256_ps128(x);
      return std::make_pair(_mm256_cvtps_pd(lo), _mm256_cvtps_pd(hi));
    }
    static_assert(!IsSame<T, double>(), "Nothing to expand to");

    if constexpr (!IsSameAny<T, float, double>()) {
      __m128i hi = _mm_srli_si128(x, 8);
      __m128i lo = x;

      if constexpr (IsSame<T, int8_t>()) {
        return std::make_pair(_mm_cvtepi8_epi16(lo), _mm_cvtepi8_epi16(hi));
      }
      if constexpr (IsSame<T, int16_t>()) {
        return std::make_pair(_mm_cvtepi16_epi32(lo), _mm_cvtepi16_epi32(hi));
      }
      if constexpr (IsSame<T, int32_t>()) {
        return std::make_pair(_mm_cvtepi32_epi64(lo), _mm_cvtepi32_epi64(hi));
      }
      static_assert(!IsSame<T, int64_t>(), "Nothing to expand to");

      if constexpr (IsSame<T, uint8_t>()) {
        return std::make_pair(_mm_cvtepu8_epi16(lo), _mm_cvtepu8_epi16(hi));
      }
      if constexpr (IsSame<T, uint16_t>()) {
        return std::make_pair(_mm_cvtepu16_epi32(lo), _mm_cvtepu16_epi32(hi));
      }
      if constexpr (IsSame<T, uint32_t>()) {
        return std::make_pair(_mm_cvtepu32_epi64(lo), _mm_cvtepu32_epi64(hi));
      }
      static_assert(!IsSame<T, uint64_t>(), "Nothing to expand to");
    }
  }

  template <typename ValidateT>
  SCANN_AVX1_INLINE ExpandsTo ExpandTo() const {
    static_assert(IsSame<ValidateT, ExpansionType>());
    const Avx1& me = *this;
    Avx1<ExpansionType, 2 * kNumRegisters> ret;
    for (size_t j : Seq(kNumRegisters)) {
      pair<ExpansionIntelType, ExpansionIntelType> expanded =
          ExpandOneRegister(*me[j]);
      ret[2 * j + 0] = expanded.first;
      ret[2 * j + 1] = expanded.second;
    }
    return ret;
  }

 private:
  std::conditional_t<kNumRegisters == 1, IntelType, Avx1<T, 1>>
      registers_[kNumRegisters];

  template <typename U, size_t kOther, size_t... kTensorOther>
  friend class Avx1;
};

template <typename T, size_t kTensorNumRegisters0,
          size_t... kTensorNumRegisters>
class Avx1 {
 public:
  using SimdSubArray = Avx1<T, kTensorNumRegisters...>;

  Avx1(Avx1Uninitialized) {}
  Avx1() : Avx1(Avx1Uninitialized()) {}

  SCANN_AVX1_INLINE Avx1(Avx1Zeros) {
    for (size_t j : Seq(kTensorNumRegisters0)) {
      tensor_[j] = Avx1Zeros();
    }
  }

  SCANN_AVX1_INLINE SimdSubArray& operator[](size_t idx) {
    DCHECK_LT(idx, kTensorNumRegisters0);
    return tensor_[idx];
  }

  SCANN_AVX1_INLINE const SimdSubArray& operator[](size_t idx) const {
    DCHECK_LT(idx, kTensorNumRegisters0);
    return tensor_[idx];
  }

  SCANN_AVX1_INLINE void Store(T* address) const {
    constexpr size_t kStride =
        sizeof(decltype(SimdSubArray().Store())) / sizeof(T);
    for (size_t j : Seq(kTensorNumRegisters0)) {
      tensor_[j].Store(address + j * kStride);
    }
  }

  using StoreResultType =
      array<decltype(SimdSubArray().Store()), kTensorNumRegisters0>;
  SCANN_AVX1_INLINE StoreResultType Store() const {
    StoreResultType ret;
    for (size_t j : Seq(kTensorNumRegisters0)) {
      ret[j] = tensor_[j].Store();
    }
    return ret;
  }

 private:
  SimdSubArray tensor_[kTensorNumRegisters0];
};

template <typename T, size_t... kNumRegisters>
SCANN_AVX1_INLINE Avx1<T, index_sequence_sum_v<kNumRegisters...>> Avx1Concat(
    const Avx1<T, kNumRegisters>&... inputs) {
  Avx1<T, index_sequence_sum_v<kNumRegisters...>> ret;

  size_t idx = 0;
  auto assign_one_input = [&](auto input) SCANN_AVX1_INLINE_LAMBDA {
    for (size_t jj : Seq(decltype(input)::kNumRegisters)) {
      ret[idx++] = input[jj];
    }
  };
  (assign_one_input(inputs), ...);

  return ret;
}

template <typename T, typename AllButLastSeq, size_t kLast>
struct Avx1ForImpl;

template <typename T, size_t... kAllButLast, size_t kLast>
struct Avx1ForImpl<T, index_sequence<kAllButLast...>, kLast> {
  using type = Avx1<T, kAllButLast..., avx1::InferNumRegisters<T, kLast>()>;
};

template <typename T, size_t... kTensorNumElements>
using Avx1For =
    typename Avx1ForImpl<T,
                         index_sequence_all_but_last_t<kTensorNumElements...>,
                         index_sequence_last_v<kTensorNumElements...>>::type;

static_assert(IsSame<Avx1For<uint8_t, 16>, Avx1<uint8_t>>());
static_assert(IsSame<Avx1For<uint8_t, 16>, Avx1<uint8_t, 1>>());
static_assert(IsSame<Avx1For<uint8_t, 32>, Avx1<uint8_t, 2>>());
static_assert(IsSame<Avx1For<uint64_t, 32>, Avx1<uint64_t, 16>>());

SCANN_AVX1_INLINE uint32_t GetComparisonMask(Avx1<int16_t> a, Avx1<int16_t> b) {
  return _mm_movemask_epi8(_mm_packs_epi16(*a, *b));
}

SCANN_AVX1_INLINE uint32_t GetComparisonMask(Avx1<int16_t, 2> cmp) {
  return GetComparisonMask(cmp[0], cmp[1]);
}

SCANN_AVX1_INLINE uint32_t GetComparisonMask(Avx1<int16_t, 4> cmp) {
  const uint32_t m00 = GetComparisonMask(cmp[0], cmp[1]);
  const uint32_t m16 = GetComparisonMask(cmp[2], cmp[3]);
  return m00 + (m16 << 16);
}

SCANN_AVX1_INLINE uint32_t GetComparisonMask(Avx1<int16_t> cmp[2]) {
  return GetComparisonMask(cmp[0], cmp[1]);
}

SCANN_AVX1_INLINE uint32_t GetComparisonMask(Avx1<float> v00, Avx1<float> v08,
                                             Avx1<float> v16, Avx1<float> v24) {
  const uint32_t m00 = _mm256_movemask_ps(*v00);
  const uint32_t m08 = _mm256_movemask_ps(*v08);
  const uint32_t m16 = _mm256_movemask_ps(*v16);
  const uint32_t m24 = _mm256_movemask_ps(*v24);
  return m00 + (m08 << 8) + (m16 << 16) + (m24 << 24);
}

SCANN_AVX1_INLINE uint32_t GetComparisonMask(Avx1<float, 4> cmp) {
  return GetComparisonMask(cmp[0], cmp[1], cmp[2], cmp[3]);
}

namespace avx1 {

SCANN_INLINE string_view SimdName() { return "AVX1"; }
SCANN_INLINE bool RuntimeSupportsSimd() { return RuntimeSupportsAvx1(); }

template <typename T, size_t... kTensorNumRegisters>
using Simd = Avx1<T, kTensorNumRegisters...>;

template <typename T, size_t kTensorNumElements0, size_t... kTensorNumElements>
using SimdFor = Avx1For<T, kTensorNumElements0, kTensorNumElements...>;

using Zeros = Avx1Zeros;
using Uninitialized = Avx1Uninitialized;

}  // namespace avx1
}  // namespace research_scann

#else

namespace research_scann {

template <typename T, size_t... kTensorNumRegisters>
struct Avx1;

}

#endif
#endif
