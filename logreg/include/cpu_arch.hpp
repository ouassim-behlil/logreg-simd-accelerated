#ifndef CPU_ARCH_H
# define CPU_ARCH_H

// Default all CPU flags to 0 
#define CPU_X86 0 
#define CPU_ARM 0 
#define CPU_UNKNOWN 0

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
	#undef CPU_X86
    #define CPU_X86 1
#elif defined(__aarch64__) || defined(__arm__) || defined(_M_ARM) || defined(_M_ARM64)
	#undef CPU_ARM
    #define CPU_ARM 1
#else
	#undef CPU_UNKNOWN
    #define CPU_UNKNOWN 1
#endif

#endif