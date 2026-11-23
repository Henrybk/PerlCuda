# PerlCuda

**Standalone Perl + C++ + CUDA batch benchmark.** No OpenKore dependencies.

The workload is **1,000 independent block-distance calculations per batch**. Distance is defined as the Chebyshev distance on an 8-direction grid:

    blockDistance((x1,y1),(x2,y2)) = max(abs(x2-x1), abs(y2-y1))

All five implementations use identical deterministic inputs, check each result before timing, and produce a checksum. This evaluates end-to-end overhead; for such cheap arithmetic, the GPU may be slower than the CPU.

## Five implementation scripts

| Script | Backend | What it includes |
| --- | --- | --- |
| 01_perl_scalar.pl | Pure Perl | One Perl subroutine call per distance |
| 02_perl_batch.pl | Pure Perl | One call for the whole batch; inline arithmetic |
| 03_cpp_ffi.pl | Native C++ | FFI::Platypus array marshaling plus native loop |
| 04_cuda_ffi.pl | CUDA | Fresh device allocation, host/device transfers, kernel, release |
| 05_cuda_cached_ffi.pl | CUDA | Reusable device buffers (transfers still happen each batch) |

The C-compatible native functions receive four input int32 arrays, an output int32 array and the element count. They return 0 on success. CUDA errors are reported, not silently ignored.

## Windows setup

Requirements:

- 64-bit Strawberry Perl and FFI::Platypus (install with cpanm FFI::Platypus)
- Visual Studio Build Tools with the MSVC x64 C++ toolchain
- NVIDIA CUDA Toolkit, compatible driver and GPU for the CUDA variants

Open the **x64 Native Tools Command Prompt for Visual Studio** in the repository directory and run:

    cpanm FFI::Platypus
    powershell -NoProfile -File .\build.ps1 -RequireCuda
    perl .\test_correctness.pl
    perl .\bench_all.pl --size 1000 --repeats 1000

If you do not have the CUDA Toolkit installed, omit -RequireCuda. The build script compiles C++ and skips CUDA. The all-in-one runner skips native variants if their DLL or FFI module is absent.

To benchmark one implementation:

    perl .\01_perl_scalar.pl --size 1000 --repeats 1000
    perl .\02_perl_batch.pl --size 1000 --repeats 1000
    perl .\03_cpp_ffi.pl --size 1000 --repeats 1000
    perl .\04_cuda_ffi.pl --size 1000 --repeats 1000
    perl .\05_cuda_cached_ffi.pl --size 1000 --repeats 1000

The native DLLs are generated inside build/ by build.ps1. They are not committed to Git.

## Linux (optional)

    cpanm FFI::Platypus
    bash build.sh
    perl test_correctness.pl
    perl bench_all.pl --size 1000 --repeats 1000

build.sh builds the native C++ library and, if nvcc is installed, the CUDA library.

## Measuring fairly

The combined runner prints CSV with these columns:

    method,size,repeats,elapsed_seconds,microseconds_per_batch,nanoseconds_per_distance,checksum

The time **includes** Perl-to-FFI conversion, host/device copies, kernel completion, and return to Perl, but **excludes** library loading and five warm-up calls. It is wall-clock elapsed time, not exclusive process CPU time or GPU-only kernel time.

For comparative data, run several times on the same machine with no competing load. Test batch sizes of 1, 100, 1000, 10000 and 100000, looking for the batch-size crossover. A separate GPU-event benchmark would be necessary to report kernel-only GPU timing.

The correctness suite checks output parity for batch lengths 1, 7, 1000 and 1025, negative coordinates, zero distance and altered inputs. CUDA correctness is tested when its shared library is present; a runtime CUDA failure is an error.

## Design limitations

- Fixed, sequential batch workflow; not an asynchronous GPU pipeline.
- Reused CUDA buffers are process-global and **not thread-safe**.
- GPU buffers are cached, but input and output transfers are still performed every call.
- Coordinate input ranges intentionally avoid signed 32-bit overflow.
- Outputs are exact integers; no approximate or floating-point shortcuts.
- No CUDA or OpenKore-specific behavior or data files are required to use the Perl and C++ examples.
- A win for the GPU here would not prove that it improves A* or other irregular workloads; benchmark those separately.

## License

MIT License. See LICENSE.
