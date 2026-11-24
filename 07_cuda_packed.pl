#!/usr/bin/env perl
use strict;
use warnings;
use FindBin;
use lib "$FindBin::Bin/lib";
use PerlCuda::Packed qw(run_packed_benchmark);
# Reuses the same CUDA cached native kernel and device buffers as benchmark 05.
run_packed_benchmark('blockdist_cuda', 'blockdist_cuda_cached', 'cuda_cached', 'prepacked', $FindBin::Bin);
