#include <cstdlib>
#include <cstdint>

float* aligned_alloc_float(size_t n, size_t alignment) {
    void* ptr = nullptr;

#if defined(_MSC_VER)
    // Windows (MSVC)
    ptr = _aligned_malloc(n * sizeof(float), alignment);
    if (!ptr) return nullptr;

#elif defined(__MINGW32__)
    // MinGW also supports _aligned_malloc
    ptr = _aligned_malloc(n * sizeof(float), alignment);
    if (!ptr) return nullptr;

#else
    // POSIX (Linux, macOS, BSD)
    if (posix_memalign(&ptr, alignment, n * sizeof(float)) != 0)
        return nullptr;
#endif

    return (float*)ptr;
}

void aligned_free_float(void* ptr) {
#if defined(_MSC_VER) || defined(__MINGW32__)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}
