#include <cuda_runtime.h>
#include <cstdint>
#if defined(_WIN32)
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

// Single-process cache, intentionally NOT thread-safe.
struct DeviceBuffers {
    int32_t *x1 = nullptr, *y1 = nullptr, *x2 = nullptr, *y2 = nullptr, *out = nullptr;
    int32_t capacity = 0;
};
static DeviceBuffers cached;

__global__ void blockdist_kernel(const int32_t* x1, const int32_t* y1,
                                 const int32_t* x2, const int32_t* y2,
                                 int32_t* out, int32_t n) {
    int32_t i = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    int32_t dx = x2[i] - x1[i], dy = y2[i] - y1[i];
    if (dx < 0) dx = -dx;
    if (dy < 0) dy = -dy;
    out[i] = dx > dy ? dx : dy;
}

static void release(DeviceBuffers& b) {
    if (b.x1) cudaFree(b.x1);
    if (b.y1) cudaFree(b.y1);
    if (b.x2) cudaFree(b.x2);
    if (b.y2) cudaFree(b.y2);
    if (b.out) cudaFree(b.out);
    b = DeviceBuffers{};
}

static cudaError_t allocate(DeviceBuffers& b, int32_t n) {
    const size_t bytes = static_cast<size_t>(n) * sizeof(int32_t);
    cudaError_t e;
    if ((e = cudaMalloc(reinterpret_cast<void**>(&b.x1), bytes)) != cudaSuccess) return e;
    if ((e = cudaMalloc(reinterpret_cast<void**>(&b.y1), bytes)) != cudaSuccess) return e;
    if ((e = cudaMalloc(reinterpret_cast<void**>(&b.x2), bytes)) != cudaSuccess) return e;
    if ((e = cudaMalloc(reinterpret_cast<void**>(&b.y2), bytes)) != cudaSuccess) return e;
    if ((e = cudaMalloc(reinterpret_cast<void**>(&b.out), bytes)) != cudaSuccess) return e;
    b.capacity = n;
    return cudaSuccess;
}

static int execute(const DeviceBuffers& b, const int32_t* x1, const int32_t* y1,
                   const int32_t* x2, const int32_t* y2, int32_t* out, int32_t n) {
    const size_t bytes = static_cast<size_t>(n) * sizeof(int32_t);
    cudaError_t e;
    if ((e = cudaMemcpy(b.x1, x1, bytes, cudaMemcpyHostToDevice)) != cudaSuccess) return (int)e;
    if ((e = cudaMemcpy(b.y1, y1, bytes, cudaMemcpyHostToDevice)) != cudaSuccess) return (int)e;
    if ((e = cudaMemcpy(b.x2, x2, bytes, cudaMemcpyHostToDevice)) != cudaSuccess) return (int)e;
    if ((e = cudaMemcpy(b.y2, y2, bytes, cudaMemcpyHostToDevice)) != cudaSuccess) return (int)e;
    constexpr int threads = 256;
    blockdist_kernel<<<(n + threads - 1) / threads, threads>>>(b.x1, b.y1, b.x2, b.y2, b.out, n);
    if ((e = cudaGetLastError()) != cudaSuccess) return (int)e;
    // Default-stream device-to-host memcpy synchronizes completion.
    if ((e = cudaMemcpy(out, b.out, bytes, cudaMemcpyDeviceToHost)) != cudaSuccess) return (int)e;
    return 0;
}

API int blockdist_cuda(const int32_t* x1, const int32_t* y1,
                       const int32_t* x2, const int32_t* y2,
                       int32_t* out, int32_t n) {
    if (!x1 || !y1 || !x2 || !y2 || !out || n <= 0) return -1;
    DeviceBuffers local;
    const cudaError_t e = allocate(local, n);
    if (e != cudaSuccess) { release(local); return (int)e; }
    const int result = execute(local, x1, y1, x2, y2, out, n);
    release(local);
    return result;
}

API int blockdist_cuda_cached(const int32_t* x1, const int32_t* y1,
                              const int32_t* x2, const int32_t* y2,
                              int32_t* out, int32_t n) {
    if (!x1 || !y1 || !x2 || !y2 || !out || n <= 0) return -1;
    if (n > cached.capacity) {
        release(cached);
        const cudaError_t e = allocate(cached, n);
        if (e != cudaSuccess) { release(cached); return (int)e; }
    }
    return execute(cached, x1, y1, x2, y2, out, n);
}
API void blockdist_cuda_release() { release(cached); }


// Diagnostic entry point only; DO NOT use this instrumented path for regular
// speed comparisons. Host wall measures synchronous pageable-memory copies;
// the kernel is measured separately with CUDA events on the default stream.
#include <chrono>
API int blockdist_cuda_profile_cached(const int32_t* x1, const int32_t* y1,
                                      const int32_t* x2, const int32_t* y2,
                                      int32_t* out, int32_t n,
                                      float* metrics_ms) {
    if (!x1 || !y1 || !x2 || !y2 || !out || !metrics_ms || n <= 0) return -1;
    if (n > cached.capacity) {
        release(cached);
        const cudaError_t allocation = allocate(cached, n);
        if (allocation != cudaSuccess) {
            release(cached);
            return (int)allocation;
        }
    }

    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    cudaError_t e = cudaEventCreate(&start_event);
    if (e != cudaSuccess) return (int)e;
    e = cudaEventCreate(&stop_event);
    if (e != cudaSuccess) {
        cudaEventDestroy(start_event);
        return (int)e;
    }

    const size_t bytes = static_cast<size_t>(n) * sizeof(int32_t);
    using Clock = std::chrono::steady_clock;
    const auto all_start = Clock::now();
    const auto h2d_start = Clock::now();
    // Run all transfers synchronously, so the host timers include driver time.
    if ((e = cudaMemcpy(cached.x1, x1, bytes, cudaMemcpyHostToDevice)) != cudaSuccess) goto cleanup;
    if ((e = cudaMemcpy(cached.y1, y1, bytes, cudaMemcpyHostToDevice)) != cudaSuccess) goto cleanup;
    if ((e = cudaMemcpy(cached.x2, x2, bytes, cudaMemcpyHostToDevice)) != cudaSuccess) goto cleanup;
    if ((e = cudaMemcpy(cached.y2, y2, bytes, cudaMemcpyHostToDevice)) != cudaSuccess) goto cleanup;

    {
        const auto h2d_end = Clock::now();
        if ((e = cudaEventRecord(start_event)) != cudaSuccess) goto cleanup;
        constexpr int threads = 256;
        blockdist_kernel<<<(n + threads - 1) / threads, threads>>>(
            cached.x1, cached.y1, cached.x2, cached.y2, cached.out, n);
        if ((e = cudaGetLastError()) != cudaSuccess) goto cleanup;
        if ((e = cudaEventRecord(stop_event)) != cudaSuccess) goto cleanup;
        if ((e = cudaEventSynchronize(stop_event)) != cudaSuccess) goto cleanup;
        float kernel_ms = 0.0f;
        if ((e = cudaEventElapsedTime(&kernel_ms, start_event, stop_event)) != cudaSuccess) goto cleanup;

        const auto d2h_start = Clock::now();
        if ((e = cudaMemcpy(out, cached.out, bytes, cudaMemcpyDeviceToHost)) != cudaSuccess) goto cleanup;
        const auto all_end = Clock::now();
        metrics_ms[0] = std::chrono::duration<float, std::milli>(h2d_end - h2d_start).count();
        metrics_ms[1] = kernel_ms;
        metrics_ms[2] = std::chrono::duration<float, std::milli>(all_end - d2h_start).count();
        metrics_ms[3] = std::chrono::duration<float, std::milli>(all_end - all_start).count();
    }
cleanup:
    cudaEventDestroy(stop_event);
    cudaEventDestroy(start_event);
    return (int)e;
}
