#ifndef SIMD_FN_H
# define SIMD_FN_H
# include <stdint.h>
# include <xmmintrin.h>
# include <emmintrin.h>
# include <immintrin.h>
# include <cmath>

float* aligned_alloc_float(size_t n, size_t alignment);
void aligned_free_float(void* ptr);

// Dot product functions
float	dot_scalar(const float* a, const float* b, uint64_t n);
float	dot_sse(const float* a, const float* b, uint64_t n);
float	dot_avx(const float* a, const float* b, uint64_t n);
float	dot_avx2_fma(const float* a, const float* b, uint64_t n);

// Exp functions
float*	sigmoid_scalar(const float* a, uint64_t n);
float*	sigmoid_sse(const float* a, uint64_t n);
float*	sigmoid_avx(const float* a, uint64_t n);
float*	sigmoid_avx2_fma(const float* a, uint64_t n);

#endif