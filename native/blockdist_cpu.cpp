#include <cstdint>

#if defined(_WIN32)
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

// Chebyshev distance on an 8-direction grid.
API int blockdist_cpu(const int32_t* x1, const int32_t* y1,
                      const int32_t* x2, const int32_t* y2,
                      int32_t* out, int32_t n) {
    if (!x1 || !y1 || !x2 || !y2 || !out || n <= 0) return -1;
    for (int32_t i = 0; i < n; ++i) {
        int32_t dx = x2[i] - x1[i];
        int32_t dy = y2[i] - y1[i];
        if (dx < 0) dx = -dx;
        if (dy < 0) dy = -dy;
        out[i] = dx > dy ? dx : dy;
    }
    return 0;
}
