#include "include/simd_fn.hpp"

float dot_scalar(const float* a, const float* b, uint64_t n) {
	uint64_t	i{0};
	float		dot{0};

	while (i < n) {
		dot += a[i] * b[i];
		i++;
	}
	return (dot);
}

float dot_sse(const float* a, const float* b, uint64_t n) {
	alignas(16) float		temp[4];
	__m128					acc;
	uint64_t				i{0};
	__m128					va;
	__m128					vb;
	float					sum{0};

	acc = _mm_setzero_ps();
	while (i + 4 < n) {;
		va = _mm_load_ps(a + i);
		vb = _mm_load_ps(b + i);
		acc = _mm_add_ps(acc, _mm_mul_ps(va, vb));
		i += 4;
	}

	// store in an aligned temp
	_mm_store_ps(temp, acc);
	sum += temp[0] + temp[1] + temp[2] + temp[3];

	while (i < n) {
		sum += a[i] * b[i];
		i++;
	}

	return (sum);
}

float dot_avx(const float* a, const float* b, uint64_t n) {
	alignas(32) float		temp[8];
	__m256					acc;
	uint64_t				i{0};
	__m256					va;
	__m256					vb;
	float					sum{0};

	acc = _mm256_setzero_ps();
	while (i + 8 <= n) {;
		va = _mm256_load_ps(a + i);
		vb = _mm256_load_ps(b + i);
		acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
		i += 8;
	}

	// store in an aligned temp
	_mm256_store_ps(temp, acc);
	sum += temp[0] + temp[1] + temp[2] + temp[3] 
			+ temp[4] + temp[5] + temp[6] + temp[7];

	while (i < n) {
		sum += a[i] * b[i];
		i++;
	}

	return (sum);
}

float dot_avx2_fma(const float* a, const float* b, uint64_t n) {
	alignas(32) float		temp[8];
	__m256					acc;
	uint64_t				i{0};
	__m256					va;
	__m256					vb;
	float					sum{0};

	acc = _mm256_setzero_ps();
	while (i + 8 <= n) {;
		va = _mm256_load_ps(a + i);
		vb = _mm256_load_ps(b + i);
		acc = _mm256_fmadd_ps(va, vb, acc); // acc+= va * vb
		i += 8;
	}

	// store in an aligned temp
	_mm256_store_ps(temp, acc);
	sum += temp[0] + temp[1] + temp[2] + temp[3] 
			+ temp[4] + temp[5] + temp[6] + temp[7];

	while (i < n) {
		sum += a[i] * b[i];
		i++;
	}

	return (sum);
}