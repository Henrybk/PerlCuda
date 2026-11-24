# PerlCuda

Standalone **Perl vs C++ vs CUDA** benchmark for batched Chebyshev (block) distance. No OpenKore dependencies.

Distance: `max(abs(x2-x1), abs(y2-y1))`. All implementations use deterministic integer coordinates and must match the Perl reference. The default is 1,000 distances per batch.

## Backends

| Script | Implementation | What is measured |
| --- | --- | --- |
| `01_perl_scalar.pl` | Perl scalar | One Perl function call per distance |
| `02_perl_batch.pl` | Perl batched | Inline Perl loop |
| `03_cpp_ffi.pl` | C++ FFI arrays | Converts five Perl arrays each call |
| `04_cuda_ffi.pl` | CUDA fresh | FFI arrays + GPU allocation + copies + kernel |
| `05_cuda_cached_ffi.pl` | CUDA cached | FFI arrays + copies + kernel, device buffers reused |
| `06_cpp_packed.pl` | C++ packed | Packed Perl scalar buffers, native pointers |
| `07_cuda_packed.pl` | CUDA packed | Packed buffers + cached GPU allocations |
| `08_cuda_profile.pl` | CUDA diagnostics | Separate transfer, kernel and instrumented total timings |

The first five benchmarks remain available unchanged.

**Packed modes** (`06` and `07`):

- `--mode prepacked` (default): pack four input arrays and allocate output once, before timing. Pass raw pointers without `sint32[]` marshaling. The timed loop includes FFI, native computation, CUDA transfers if applicable, and an O(1) checksum read per call.
- `--mode end-to-end`: repack all four input arrays, obtain pointers, execute, and unpack all results on *every* iteration. Measures realistic Perl-array preparation plus native execution.

The binary representation is little-endian, signed int32 (`pack('l<*', ...)`). Prepacked buffers remain alive and unchanged throughout each synchronous native call. CUDA cached buffers are process-global and **not thread-safe**.

## Windows setup

Use **64-bit Strawberry Perl**, `FFI::Platypus` (with its Buffer module), the MSVC x64 toolchain, and NVIDIA CUDA Toolkit for GPU runs. In the **x64 Native Tools Command Prompt**:

```cmd
cd /d K:\Github\PerlCuda
git pull
cpanm FFI::Platypus
powershell -NoProfile -ExecutionPolicy Bypass -File .\build.ps1 -RequireCuda
perl test_correctness.pl
perl test_packed.pl
perl bench_all.pl --size 1000 --repeats 1000
```

If the repo is elsewhere, change the directory. If CUDA is unavailable, omit `-RequireCuda` to build only C++; the combined runner skips missing optional DLLs. GPU runtime errors are not silently ignored.

## Run separate modes

```cmd
perl 06_cpp_packed.pl --size 10000 --repeats 1000 --mode prepacked
perl 07_cuda_packed.pl --size 10000 --repeats 1000 --mode prepacked
perl 06_cpp_packed.pl --size 10000 --repeats 1000 --mode end-to-end
perl 07_cuda_packed.pl --size 10000 --repeats 1000 --mode end-to-end
perl 08_cuda_profile.pl --size 10000 --repeats 200
```

The combined `bench_all.pl` includes both new packed modes along with the five old methods; it prints CSV with `method,size,repeats,elapsed_seconds,microseconds_per_batch,nanoseconds_per_distance,checksum`.

For comparisons:

```cmd
perl bench_all.pl --size 1000 --repeats 1000
perl bench_all.pl --size 10000 --repeats 1000
perl bench_all.pl --size 100000 --repeats 100
```

## GPU profiling methodology

`08_cuda_profile.pl` uses a **separate instrumented CUDA function**. It measures:

- **H2D:** host steady-clock wall time for four synchronous pageable-memory transfers.
- **Kernel:** CUDA event elapsed GPU time between events recorded around the kernel.
- **D2H:** host steady-clock wall time for synchronous output transfer.
- **Native wall:** whole instrumented function, including event synchronization.
- **Perl wall:** FFI calls plus timing reads.

CUDA event creation and synchronization add overhead; these metrics are diagnostic, **not** a decomposition of an uninstrumented benchmark and **not** process CPU time. Uninstrumented `07_cuda_packed.pl` is the appropriate primary performance comparison.

## Earlier baseline: FFI array-marshaling methods

These are the original user-measured *wall microseconds per batch*, **not** results of the new implementations.

| Backend | 1,000 | 10,000 | 100,000 |
| --- | ---: | ---: | ---: |
| Perl scalar | 249.9 | 2495.9 | 25552.0 |
| Perl batch | **171.0** | **1676.4** | **16900.1** |
| C++ FFI arrays | 194.4 | 1990.1 | 19578.5 |
| CUDA fresh | 664.4 | 2458.3 | 20220.7 |
| CUDA cached | 309.4 | 2103.2 | 20278.1 |

The earlier C++ and CUDA FFI calls converted five full Perl arrays on each invocation. The new modes isolate that marshaling cost.

## Linux (optional)

```sh
cpanm FFI::Platypus
bash build.sh
perl test_correctness.pl
perl test_packed.pl
perl bench_all.pl --size 1000 --repeats 1000
```

## Limitations

This is a synchronous, non-pipelined GPU experiment on little-endian x64. Input coordinate ranges avoid signed integer overflow. Packed buffers are ordinary Perl string memory, not CUDA-pinned buffers. Runtime CUDA device execution and performance must be verified on an NVIDIA-enabled machine; compilation alone is insufficient. These results do not demonstrate acceleration of A* or other irregular workloads.

MIT licensed; see [LICENSE](LICENSE).
