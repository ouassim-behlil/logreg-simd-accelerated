#include "include/simd_fn.hpp"


float*	sigmoid_scalar(const float* a, uint64_t n) {
	uint64_t		i{0};
	float*			out;

	out = aligned_alloc_float(n, 16);
	if (!out) { return (nullptr); }
	while (i < n) {
		out[i] = 1.0f / (1.0f + std::exp(-a[i]));
		i++;
	}
	return (out);
}


// ============================================================
//  SSE2  (128-bit, 4 floats)
// ============================================================

// Horner evaluation of exp(r) for |r| <= ln2/2
// exp(r) ≈ 1 + r*(1 + r*(1/2 + r*(1/6 + r*(1/24 + r*(1/120)))))
static inline __m128	exp_poly_sse(__m128 r)
{
	// Horner coefficients from innermost to outermost
	const __m128 c5 = _mm_set1_ps(1.0f / 120.0f);
	const __m128 c4 = _mm_set1_ps(1.0f / 24.0f);
	const __m128 c3 = _mm_set1_ps(1.0f / 6.0f);
	const __m128 c2 = _mm_set1_ps(0.5f);
	const __m128 c1 = _mm_set1_ps(1.0f);
	const __m128 c0 = _mm_set1_ps(1.0f);

	// p = c5
	__m128 p = c5;
	// p = c4 + r * p
	p = _mm_add_ps(c4, _mm_mul_ps(r, p));
	// p = c3 + r * p
	p = _mm_add_ps(c3, _mm_mul_ps(r, p));
	// p = c2 + r * p
	p = _mm_add_ps(c2, _mm_mul_ps(r, p));
	// p = c1 + r * p
	p = _mm_add_ps(c1, _mm_mul_ps(r, p));
	// p = c0 + r * p
	p = _mm_add_ps(c0, _mm_mul_ps(r, p));

	return (p);
}

static inline __m128	vector_exp_sse(__m128 v)
{
	const __m128 LOG2E = _mm_set1_ps(1.44269504088896341f);
	const __m128 LN2   = _mm_set1_ps(0.69314718055994531f);

	// n = round(v * log2(e))  — portable: uses MXCSR round-to-nearest
	__m128  y   = _mm_mul_ps(v, LOG2E);
	__m128i n_i = _mm_cvtps_epi32(y);          // round to nearest int (SSE2)
	__m128  n   = _mm_cvtepi32_ps(n_i);        // back to float

	// r = v - n * ln2  (range reduction)
	__m128 r = _mm_sub_ps(v, _mm_mul_ps(n, LN2));

	// exp(r) via Horner scheme
	__m128 er = exp_poly_sse(r);

	// 2^n via IEEE 754 bit manipulation: (n + 127) << 23
	__m128i exp_bits = _mm_slli_epi32(_mm_add_epi32(n_i, _mm_set1_epi32(127)), 23);
	__m128  two_n    = _mm_castsi128_ps(exp_bits);

	// exp(v) = exp(r) * 2^n
	return (_mm_mul_ps(er, two_n));
}

static inline __m128	vect_sigmoid_sse(__m128 v)
{
	const __m128 one = _mm_set1_ps(1.0f);
	__m128 neg_v = _mm_sub_ps(_mm_setzero_ps(), v);
	__m128 e     = vector_exp_sse(neg_v);
	return (_mm_div_ps(one, _mm_add_ps(one, e)));
}

float*	sigmoid_sse(const float* a, uint64_t n)
{
	float*   out;
	uint64_t i{0};

	out = aligned_alloc_float((size_t)n, 16);
	if (!out) { return (nullptr); }

	while (i + 4 <= n) {
		__m128 x     = _mm_load_ps(a + i);
		__m128 sig_x = vect_sigmoid_sse(x);
		_mm_store_ps(out + i, sig_x);
		i += 4;
	}
	while (i < n) {
		out[i] = 1.0f / (1.0f + std::exp(-a[i]));
		i++;
	}
	return (out);
}

// ============================================================
//  AVX  (256-bit, 8 floats)
// ============================================================

// Horner evaluation of exp(r) for |r| <= ln2/2  — AVX version
static inline __m256	exp_poly_avx(__m256 r)
{
	const __m256 c5 = _mm256_set1_ps(1.0f / 120.0f);
	const __m256 c4 = _mm256_set1_ps(1.0f / 24.0f);
	const __m256 c3 = _mm256_set1_ps(1.0f / 6.0f);
	const __m256 c2 = _mm256_set1_ps(0.5f);
	const __m256 c1 = _mm256_set1_ps(1.0f);
	const __m256 c0 = _mm256_set1_ps(1.0f);

	__m256 p = c5;
	p = _mm256_add_ps(c4, _mm256_mul_ps(r, p));
	p = _mm256_add_ps(c3, _mm256_mul_ps(r, p));
	p = _mm256_add_ps(c2, _mm256_mul_ps(r, p));
	p = _mm256_add_ps(c1, _mm256_mul_ps(r, p));
	p = _mm256_add_ps(c0, _mm256_mul_ps(r, p));

	return (p);
}

static inline __m256	vector_exp_avx(__m256 v)
{
	const __m256 LOG2E = _mm256_set1_ps(1.44269504088896341f);
	const __m256 LN2   = _mm256_set1_ps(0.69314718055994531f);

	// Portable round-to-nearest: _mm256_round_ps with _MM_FROUND_TO_NEAREST_INT
	// Available since AVX (no AVX2 needed).
	__m256  y   = _mm256_mul_ps(v, LOG2E);
	__m256  n   = _mm256_round_ps(y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	__m256i n_i = _mm256_cvtps_epi32(n);       // for 2^n bit trick

	// r = v - n * ln2
	__m256 r = _mm256_sub_ps(v, _mm256_mul_ps(n, LN2));

	// exp(r) via Horner approximation
	__m256 er = exp_poly_avx(r);

	// 2^n via bit manipulation
	__m256i exp_bits = _mm256_slli_epi32(_mm256_add_epi32(n_i, _mm256_set1_epi32(127)), 23);
	__m256  two_n    = _mm256_castsi256_ps(exp_bits);

	return (_mm256_mul_ps(er, two_n));
}

static inline __m256	vect_sigmoid_avx(__m256 v)
{
	const __m256 one = _mm256_set1_ps(1.0f);
	__m256 neg_v = _mm256_sub_ps(_mm256_setzero_ps(), v);
	__m256 e     = vector_exp_avx(neg_v);
	return (_mm256_div_ps(one, _mm256_add_ps(one, e)));
}

float*	sigmoid_avx(const float* a, uint64_t n)
{
	float*   out;
	uint64_t i{0};

	out = aligned_alloc_float((size_t)n, 32);
	if (!out) { return (nullptr); }

	while (i + 8 <= n) {
		__m256 x     = _mm256_load_ps(a + i);
		__m256 sig_x = vect_sigmoid_avx(x);
		_mm256_store_ps(out + i, sig_x);
		i += 8;
	}
	// tail: reuse SSE path (4 at a time)
	while (i + 4 <= n) {
		__m128 x     = _mm_load_ps(a + i);
		__m128 sig_x = vect_sigmoid_sse(x);
		_mm_store_ps(out + i, sig_x);
		i += 4;
	}
	while (i < n) {
		out[i] = 1.0f / (1.0f + std::exp(-a[i]));
		i++;
	}
	return (out);
}

// ============================================================
//  AVX2 + FMA  (256-bit, 8 floats, fused multiply-add)
// ============================================================

// Horner evaluation using FMA: p = fma(r, p, c)  i.e.  r*p + c
// _mm256_fmadd_ps(a, b, c) = a*b + c
static inline __m256	exp_poly_avx2_fma(__m256 r)
{
	const __m256 c5 = _mm256_set1_ps(1.0f / 120.0f);
	const __m256 c4 = _mm256_set1_ps(1.0f / 24.0f);
	const __m256 c3 = _mm256_set1_ps(1.0f / 6.0f);
	const __m256 c2 = _mm256_set1_ps(0.5f);
	const __m256 c1 = _mm256_set1_ps(1.0f);
	const __m256 c0 = _mm256_set1_ps(1.0f);

	// p = fma(r, c5, c4)  →  r*c5 + c4
	__m256 p = _mm256_fmadd_ps(r, c5, c4);
	p = _mm256_fmadd_ps(r, p, c3);
	p = _mm256_fmadd_ps(r, p, c2);
	p = _mm256_fmadd_ps(r, p, c1);
	p = _mm256_fmadd_ps(r, p, c0);

	return (p);
}

static inline __m256	vector_exp_avx2_fma(__m256 v)
{
	const __m256 LOG2E = _mm256_set1_ps(1.44269504088896341f);
	const __m256 LN2   = _mm256_set1_ps(0.69314718055994531f);

	// Portable round-to-nearest via _mm256_round_ps (AVX)
	__m256  y   = _mm256_mul_ps(v, LOG2E);
	__m256  n   = _mm256_round_ps(y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	__m256i n_i = _mm256_cvtps_epi32(n);

	// r = v - n * ln2  (use FMA for accuracy: r = fma(-n, ln2, v))
	__m256 r = _mm256_fnmadd_ps(n, LN2, v);

	// exp(r) via Horner + FMA
	__m256 er = exp_poly_avx2_fma(r);

	// 2^n via IEEE 754 bit manipulation
	__m256i exp_bits = _mm256_slli_epi32(_mm256_add_epi32(n_i, _mm256_set1_epi32(127)), 23);
	__m256  two_n    = _mm256_castsi256_ps(exp_bits);

	return (_mm256_mul_ps(er, two_n));
}

static inline __m256	vect_sigmoid_avx2_fma(__m256 v)
{
	const __m256 one = _mm256_set1_ps(1.0f);
	__m256 neg_v = _mm256_sub_ps(_mm256_setzero_ps(), v);
	__m256 e     = vector_exp_avx2_fma(neg_v);
	// 1 / (1 + e)
	return (_mm256_div_ps(one, _mm256_add_ps(one, e)));
}

float*	sigmoid_avx2_fma(const float* a, uint64_t n)
{
	float*   out;
	uint64_t i{0};

	out = aligned_alloc_float((size_t)n, 32);
	if (!out) { return (nullptr); }

	while (i + 8 <= n) {
		__m256 x     = _mm256_load_ps(a + i);
		__m256 sig_x = vect_sigmoid_avx2_fma(x);
		_mm256_store_ps(out + i, sig_x);
		i += 8;
	}
	// tail: reuse SSE path (4 at a time)
	while (i + 4 <= n) {
		__m128 x     = _mm_load_ps(a + i);
		__m128 sig_x = vect_sigmoid_sse(x);
		_mm_store_ps(out + i, sig_x);
		i += 4;
	}
	while (i < n) {
		out[i] = 1.0f / (1.0f + std::exp(-a[i]));
		i++;
	}
	return (out);
}