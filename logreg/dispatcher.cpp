#include "include/logreg_dispatcher.hpp"
#include "include/simd_fn.hpp"
#include <iostream>

// Definition of the global kernel function pointers
float  (*dot_product)(const float* a, const float* b, uint64_t n) = nullptr;
float* (*sigmoid)(const float* a, uint64_t n)                     = nullptr;

void	init_kernels()
{
	// ---- dot product ----
	if (has_avx2() && has_fma()) {
		dot_product = dot_avx2_fma;
		std::cout << "[dispatcher] dot_product : AVX2 + FMA\n";
	}
	else if (has_avx()) {
		dot_product = dot_avx;
		std::cout << "[dispatcher] dot_product : AVX\n";
	}
	else if (has_sse()) {
		dot_product = dot_sse;
		std::cout << "[dispatcher] dot_product : SSE\n";
	}
	else {
		dot_product = dot_scalar;
		std::cout << "[dispatcher] dot_product : scalar\n";
	}

	// ---- sigmoid ----
	if (has_avx2() && has_fma()) {
		sigmoid = sigmoid_avx2_fma;
		std::cout << "[dispatcher] sigmoid      : AVX2 + FMA\n";
	}
	else if (has_avx()) {
		sigmoid = sigmoid_avx;
		std::cout << "[dispatcher] sigmoid      : AVX\n";
	}
	else if (has_sse()) {
		sigmoid = sigmoid_sse;
		std::cout << "[dispatcher] sigmoid      : SSE\n";
	}
	else {
		sigmoid = sigmoid_scalar;
		std::cout << "[dispatcher] sigmoid      : scalar\n";
	}
}