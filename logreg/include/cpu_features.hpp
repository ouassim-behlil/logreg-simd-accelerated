// cpu_features.hpp file

#ifndef CPU_FEATURES_H
# define CPU_FEATURES_H

#include "cpu_arch.hpp"
#include <stdint.h>

// Detect sse
#if CPU_X86

// Microsoft visual c++ version
#if defined(_MSC_VER)
	#include <intrin.h>

	static inline void	cpuid(int out[4], int leaf){
		__cpuidex(out, leaf, 0);
	}

	static inline uint64_t	xgetbv(uint32_t index) {
		return _xgetbv(index);
	}
	
// GCC/CLANG version
#else
	#include <cpuid.h>
	
	static inline void	cpuid(int out[4], int leaf){
		__cpuid_count(leaf, 0, out[0], out[1], out[2], out[3]);
	}

	static inline uint64_t	xgetbv(uint32_t index) {
		uint32_t	eax;
		uint32_t	edx;

		__asm__ volatile ("xgetbv" :  "=a"(eax), "=d"(edx): "c"(index));
		return (((uint64_t)edx << 32) | eax);
	}
# endif // _MSC_VER

static inline bool	has_sse() {
	int		result[4];

	cpuid(result, 1);
	return (result[3] & (1 << 25)); // edx bit 25 = sse
}

static inline bool	has_sse2() {
	int		result[4];

	cpuid(result, 1);
	return (result[3] & (1 << 26)); //edx bit 26 = sse2
}

static inline bool	has_avx() {
	int			result[4];
	bool		cpu_avx;
	bool		cpu_xsave;
	bool		os_avx;
	uint64_t	xcr0;

	cpuid(result, 1);
	cpu_avx = result[2] & (1 << 28);
	cpu_xsave = result[2] & (1 << 27);

	if (!cpu_avx || !cpu_xsave) return (false);

	xcr0 = xgetbv(0);

	os_avx = (xcr0 & 0x6) == 0x6;

	return (os_avx);
}

static inline bool	has_avx2() {
	int		result[4];

	if (!has_avx()) { return (false); }

	cpuid(result, 7);
	return (result[1] & (1 << 5));
}

static inline bool	has_fma() {
	int		result[4];

	cpuid(result, 1);
	return (result[2] & (1 << 12));
}

# else // sse and avx doesn't exist on non x86 cpus

static inline bool	has_sse() { return (false); }
static inline bool	has_sse2() { return (false); }
static inline bool	has_avx() { return (false); }
static inline bool	has_avx2() { return (false); }
static inline bool has_fma() {
	#if defined(__aarch64__)
		return (true); // ARMv8 always has FMA 
	#elif defined(__ARM_FEATURE_FMA)
		return (true);  // ARMv7 with FMA 
	#else
		return (false);
	#endif
}
# endif // CPU_X86


#endif // CPU_FEATURES_H