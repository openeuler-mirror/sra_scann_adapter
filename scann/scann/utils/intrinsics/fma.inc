

template <typename T>
SCANN_SIMD_INLINE T FusedMultiplyAdd(T a, T b, T acc) {
  if constexpr (IsSame<T, Avx2<float>>()) {
    return _mm256_fmadd_ps(*a, *b, *acc);
  }
  if constexpr (IsSame<T, Avx2<double>>()) {
    return _mm256_fmadd_pd(*a, *b, *acc);
  }
  if constexpr (IsSame<T, Avx512<float>>()) {
    return _mm512_fmadd_ps(*a, *b, *acc);
  }
  if constexpr (IsSame<T, Sse4<float>>()) {
    return _mm_fmadd_ps_mine(*a, *b, *acc);
  }
  if constexpr (IsSame<T, Sse4<double>>()) {
    return _mm_fmadd_pd_mine(*a, *b, *acc);
  }

  return acc + (a * b);
}

template <typename T>
SCANN_SIMD_INLINE void FusedMultiplyAdd(T a, T b, T* acc) {
  *acc = FusedMultiplyAdd(a, b, *acc);
}

template <typename T>
SCANN_SIMD_INLINE T FusedMultiplySubtract(T a, T b, T acc) {
  if constexpr (IsSame<T, Avx2<float>>()) {
    return _mm256_fnmadd_ps(*a, *b, *acc);
  }
  if constexpr (IsSame<T, Avx2<double>>()) {
    return _mm256_fnmadd_pd(*a, *b, *acc);
  }
  if constexpr (IsSame<T, Avx512<float>>()) {
    return _mm512_fnmadd_ps(*a, *b, *acc);
  }
  if constexpr (IsSame<T, Avx512<double>>()) {
    return _mm512_fnmadd_pd(*a, *b, *acc);
  }
  if constexpr (IsSame<T, Sse4<float>>()) {
    return _mm_fnmadd_ps_mine(*a, *b, *acc);
  }
  if constexpr (IsSame<T, Sse4<double>>()) {
    return _mm_fnmadd_pd_mine(*a, *b, *acc);
  }

  return acc - (a * b);
}

template <typename T>
SCANN_SIMD_INLINE void FusedMultiplySubtract(T a, T b, T* acc) {
  *acc = FusedMultiplySubtract(a, b, *acc);
}
