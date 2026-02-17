#include "include/logreg_dispatcher.hpp"
#include "include/simd_fn.hpp"
#include <iostream>

// Function pointer for the selected dot product implementation
float (*dot_product)(const float* a, const float* b, uint64_t n) = nullptr;

void	init_kernels() {
	// Detect CPU features and select the best available implementation
	if (has_avx2() && has_fma()) {
		dot_product = dot_avx2_fma;
		std::cout << "Using AVX2 + FMA kernel" << std::endl;
	}
	else if (has_avx()) {
		dot_product = dot_avx;
		std::cout << "Using AVX kernel" << std::endl;
	}
	else if (has_sse()) {
		dot_product = dot_sse;
		std::cout << "Using SSE kernel" << std::endl;
	}
	else {
		dot_product = dot_scalar;
		std::cout << "Using scalar kernel (no SIMD support)" << std::endl;
	}
}