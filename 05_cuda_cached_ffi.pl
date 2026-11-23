#!/usr/bin/env perl
use strict;
use warnings;
use FindBin;
use lib "$FindBin::Bin/lib";
use PerlCuda::Common qw(run_benchmark);
use PerlCuda::Native qw(load_function);
# Reuse device allocations, but transfer data each batch.
my $worker = load_function($FindBin::Bin, 'blockdist_cuda', 'blockdist_cuda_cached');
my $release = load_function($FindBin::Bin, 'blockdist_cuda', 'blockdist_cuda_release', 'void');
run_benchmark('cuda_ffi_cached', $worker, $release);
